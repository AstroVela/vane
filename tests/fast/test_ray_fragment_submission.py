# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import threading
import time

import pytest

import vane

ray = pytest.importorskip("ray")

import vane.runners.ray.fragment_worker_submission as fragment_submission_mod
import vane.runners.ray.fragment_worker_task_control as task_control_mod
import vane.runners.ray.worker as worker_mod
import vane.runners.ray.worker_handle as worker_handle_mod
from vane.runners.common import QueryDeadlineExceeded
from vane.runners.fte.fte_config import FteWorkerAdmissionConfig
from vane.runners.ray.fragment_worker_context import fragment_id_for_task
from vane.runners.ray.fte import (
    AssignmentResult,
    FteFragmentExecution,
    FteTaskAttemptId,
    FteTaskState,
    FteWorkerReservationUnavailable,
    NodeRequirements,
    PartitionInfo,
)
from vane.runners.ray.fte_events import TaskStatusChanged, WorkerReservationCompleted
from vane.runners.ray.query_execution_graph import (
    NodeResourceAllocation,
    QueryAllocation,
    QueryExecutionGraph,
    ResourceVector,
    StageResourceSpec,
)
from vane.runners.ray.query_graph_builder import fte_stage_id_for_fragment
from vane.runners.ray.query_resource_runtime import (
    clear_query_resource_managers,
    get_query_resource_manager,
    register_query_graph,
)
from vane.runners.ray.worker_handle import RayWorkerActorHandle as _ProductionRayWorkerActorHandle


def _test_ray_node_id() -> str:
    if ray.is_initialized():
        return str(ray.get_runtime_context().get_node_id())
    return "node-a"


class RayWorkerActorHandle(_ProductionRayWorkerActorHandle):
    def __init__(
        self,
        actor_handle,
        *,
        memory_capacity_bytes,
        worker_id=None,
        node_id=None,
    ):
        super().__init__(
            actor_handle,
            memory_capacity_bytes=memory_capacity_bytes,
            worker_id=str(worker_id or f"test-worker-{id(actor_handle)}"),
            node_id=str(node_id or _test_ray_node_id()),
        )

    def record_fte_task_result_ready(self, attempt_id):
        result = super().record_fte_task_result_ready(attempt_id)
        # These unit handles do not materialize FteWorkerTaskHandle results, so
        # model immediate result adoption before scheduling the next partition.
        self.record_fte_task_terminal(attempt_id)
        return result

    def _ensure_fragment_progress_topology(self, query_id, fragment_id, fragment_plan):
        topology = {
            "schema": "pipeline_topology",
            "pipelines": [
                {
                    "pipeline_id": 1,
                    "operators": ["TABLE_SCAN"],
                    "operator_details": [{}],
                    "stage_ids": [],
                }
            ],
        }
        return fragment_submission_mod.ensure_fte_fragment_progress_topology(
            query_id,
            fragment_id,
            lambda: topology,
        )


_ORIGINAL_START_FTE_ATTEMPT_STATUS_WATCHER = RayWorkerActorHandle._start_fte_attempt_status_watcher


class _ImmediateFuture:
    def __init__(self, value):
        self._value = value

    def result(self, timeout=None):
        if timeout is None:
            return ray.get(self._value)
        return ray.get(self._value, timeout=timeout)

    def add_done_callback(self, callback):
        callback(self)


class _ImmediateObjectRef:
    def __init__(self, value):
        self._value = value

    def future(self):
        return _ImmediateFuture(self._value)


class _FakeRemoteMethod:
    def __init__(self, fn):
        self._fn = fn
        self.option_calls = []

    def options(self, **kwargs):
        self.option_calls.append(kwargs)
        return self

    def remote(self, *args, **kwargs):
        resolved_args = [arg.future().result() if isinstance(arg, _ImmediateObjectRef) else arg for arg in args]
        return _ImmediateObjectRef(self._fn(*resolved_args, **kwargs))


class _FakeActor:
    def __init__(self):
        self.register_payloads = []
        self.fragment_calls = []
        self.drop_calls = []
        self.fte_calls = []
        self.fragment_stats_calls = 0
        self.register_fragments = _FakeRemoteMethod(self._register_fragments)
        self.drop_query_fragments = _FakeRemoteMethod(self._drop_query_fragments)
        self.stats_fragments = _FakeRemoteMethod(self._stats_fragments)
        self.fte_create_task = _FakeRemoteMethod(self._fte_create_task)
        self.fte_add_splits = _FakeRemoteMethod(self._fte_add_splits)
        self.fte_no_more_splits = _FakeRemoteMethod(self._fte_no_more_splits)
        self.fte_update_task = _FakeRemoteMethod(self._fte_update_task)
        self.fte_get_task_status = _FakeRemoteMethod(self._fte_get_task_status)
        self.fte_wait_task_status = _FakeRemoteMethod(self._fte_wait_task_status)
        self.fte_wait_split_queue_has_space = _FakeRemoteMethod(self._fte_wait_split_queue_has_space)
        self.fte_get_task_info = _FakeRemoteMethod(self._fte_get_task_info)
        self.fte_ack_task_result = _FakeRemoteMethod(self._fte_ack_task_result)
        self.fte_release_task_result = _FakeRemoteMethod(self._fte_release_task_result)
        self.fte_cancel_task = _FakeRemoteMethod(self._fte_cancel_task)
        self.fte_drop_query = _FakeRemoteMethod(self._fte_drop_query)

    def _register_fragments(self, payload):
        self.register_payloads.append(payload)
        return {"registered": len(payload), "existing": 0, "total": len(payload)}

    def _drop_query_fragments(self, query_id):
        self.drop_calls.append(query_id)
        return 1

    def _stats_fragments(self):
        self.fragment_stats_calls += 1
        return {"registered_total": 2, "existing_total": 1, "lookup_hits": 3}

    def _fte_create_task(self, request):
        self.fte_calls.append(("create", request))
        return self._control_status(
            "fte_create_task",
            request["task_id"],
        )

    @staticmethod
    def _control_status(operation, task_id, *, state="RUNNING", **fields):
        return {
            "state": state,
            "task_id": task_id,
            "_fte_control_operation": operation,
            "_fte_control_applied": True,
            **fields,
        }

    def _fte_add_splits(self, task_id, source_node_id, splits, dependency=None):
        self.fte_calls.append(("add_splits", task_id, source_node_id, splits))
        return self._control_status("fte_add_splits", task_id, version=2)

    def _fte_no_more_splits(self, task_id, source_node_id, dependency=None):
        self.fte_calls.append(("no_more_splits", task_id, source_node_id))
        return self._control_status("fte_no_more_splits", task_id, version=3)

    def _fte_update_task(self, task_id, update, dependency=None):
        self.fte_calls.append(("update_task", task_id, update))
        return self._control_status("fte_update_task", task_id, version=4)

    def _fte_get_task_status(self, task_id):
        self.fte_calls.append(("get_status", task_id))
        return {"state": "FINISHED", "task_id": task_id, "version": 5}

    def _fte_wait_task_status(self, task_id, min_version=None, timeout_s=None):
        self.fte_calls.append(("wait_status", task_id, min_version, timeout_s))
        return {"state": "FINISHED", "task_id": task_id, "version": 4}

    def _fte_wait_split_queue_has_space(
        self,
        task_id,
        source_node_id=None,
        max_buffered_splits=None,
        timeout_s=None,
    ):
        self.fte_calls.append(("wait_split_queue", task_id, source_node_id, max_buffered_splits, timeout_s))
        return {"has_space": True, "buffered_splits": 0}

    def _fte_get_task_info(self, task_id):
        self.fte_calls.append(("get_info", task_id))
        return {"status": {"state": "FINISHED"}, "task_id": task_id}

    def _fte_ack_task_result(self, task_id, dependency=None):
        self.fte_calls.append(("ack", task_id, dependency))
        return self._control_status("fte_ack_task_result", task_id, state="FINISHED")

    def _fte_release_task_result(self, task_id, dependency=None):
        self.fte_calls.append(("release", task_id, dependency))
        return self._control_status("fte_release_task_result", task_id, state="FINISHED")

    def _fte_cancel_task(self, task_id, dependency=None):
        self.fte_calls.append(("cancel", task_id))
        return self._control_status("fte_cancel_task", task_id, state="CANCELED")

    def _fte_drop_query(self, query_id):
        self.fte_calls.append(("drop_query", query_id))
        return {"tasks_removed": 1, "tasks_canceled": 0, "fragments_removed": 2}


class _FakeFteTaskHandle:
    def __init__(self, task_id, worker_handle):
        self.task_id = FteTaskAttemptId.coerce(task_id)
        self.worker_handle = worker_handle
        self.worker_id = worker_handle.worker_id


class _FakeTask:
    def __init__(
        self,
        *,
        name,
        context=None,
        inputs=None,
        plan=None,
        task_context=None,
        exchange_sink_instance=None,
    ):
        self._name = name
        self._context = dict(context or {})
        query_id = str(self._context.get("query_id") or "").strip()
        node_id = str(self._context.get("node_id") or "").strip()
        if query_id and node_id:
            self._context.setdefault("resource_query_id", query_id)
            self._context.setdefault(
                "resource_stage_id",
                f"stage:{query_id}:node:{node_id}:fte",
            )
        self._inputs = inputs or {}
        self._plan = plan if plan is not None else {"plan": name}
        self._exchange_sink_instance = exchange_sink_instance
        if task_context is None:
            try:
                last_node_id = int(self._context.get("node_id", 0))
            except (TypeError, ValueError):
                last_node_id = 0
            task_context = {
                "query_idx": 0,
                "last_node_id": last_node_id,
                "task_id": 0,
                "node_ids": [last_node_id],
            }
        self._task_context = dict(task_context)
        self.plan_calls = 0

    def name(self):
        return self._name

    def context(self):
        return dict(self._context)

    def Inputs(self):
        return dict(self._inputs)

    def task_context(self):
        return dict(self._task_context)

    def plan(self):
        self.plan_calls += 1
        return self._plan

    def exchange_sink_instance(self):
        return self._exchange_sink_instance


class _InputsFailingTask(_FakeTask):
    def Inputs(self):
        raise RuntimeError("inputs exploded")


class _MissingInputsTask(_FakeTask):
    Inputs = None


class _ExchangeSinkInstanceFailingTask(_FakeTask):
    def exchange_sink_instance(self):
        raise RuntimeError("exchange sink instance exploded")


def _exchange_selector_payload(
    splits=(),
    *,
    final=False,
    partition_count=None,
    selected=None,
):
    payload = {"selected": dict(selected or {})}
    for split in splits:
        payload["selected"][str(int(split["source_partition_id"]))] = {"split": dict(split)}
    if final:
        payload["final"] = True
    if partition_count is not None:
        payload["partition_count"] = int(partition_count)
    return payload


def _register_test_query_graph(query_id, fragment_ids, *, max_concurrency=256):
    try:
        return get_query_resource_manager(query_id)
    except KeyError:
        pass
    fragment_ids = set(fragment_ids)
    fragment_ids.update(f"{query_id}:node:{node_id}" for node_id in range(129))
    fragment_ids.update(
        f"{query_id}:node:{node_id}" for node_id in ("scan", "exchange", "upstream-worker", "worker-retry")
    )
    stages = tuple(
        StageResourceSpec(
            query_id=query_id,
            stage_id=fte_stage_id_for_fragment(query_id, fragment_id),
            physical_node_id=f"node:{fragment_id.rsplit(':node:', 1)[1]}:fte",
            stage_kind="fte",
            backend="ray_worker",
            input_stage_ids=(),
            per_task=ResourceVector(cpu=1, heap_bytes=10),
            target_output_block_bytes=1,
            generator_buffer_blocks=1,
            max_concurrency=max_concurrency,
        )
        for fragment_id in sorted(fragment_ids)
    )
    graph = QueryExecutionGraph(
        query_id=query_id,
        plan_digest=f"sha256:test:{query_id}",
        stages=stages,
        terminal_stage_ids=tuple(stage.stage_id for stage in stages),
    )
    allocation_resources = ResourceVector(
        cpu=256,
        heap_bytes=2560,
        object_store_bytes=256,
    )
    manager = register_query_graph(
        graph,
        QueryAllocation(
            resources=allocation_resources,
            node_allocations=(
                NodeResourceAllocation(
                    node_id=_test_ray_node_id(),
                    resources=allocation_resources,
                ),
            ),
            actor_placements=(),
            generation=1,
        ),
    )
    for stage in stages:
        manager.update_stage_state(stage.stage_id, runnable=True)
    return manager


def _install_manual_test_fragment(query_id, node_id, *, partition_count=1):
    fragment_id = f"{query_id}:node:{node_id}"
    _register_test_query_graph(query_id, [fragment_id])
    fragment_execution = FteFragmentExecution(
        query_id,
        7,
        fragment_id=fragment_id,
        context={
            "resource_query_id": query_id,
            "resource_stage_id": f"stage:{query_id}:node:{node_id}:fte",
        },
        task_memory_bytes=10,
    )
    for partition_id in range(partition_count):
        fragment_execution.add_partition(partition_id)
    worker_handle_mod._FTE_FRAGMENT_EXECUTIONS[(query_id, fragment_id)] = fragment_execution
    return fragment_id


@pytest.fixture(autouse=True)
def _patch_ray_worker_handle_test_state(monkeypatch):
    clear_query_resource_managers()
    worker_handle_mod._stop_fte_status_watchers()
    worker_handle_mod._FTE_FRAGMENT_EXECUTION_IDS.clear()
    worker_handle_mod._FTE_QUERY_NEXT_FRAGMENT_EXECUTION_ID.clear()
    worker_handle_mod._FTE_FRAGMENT_EXECUTIONS.clear()
    worker_handle_mod._FTE_PARTITION_OWNERS.clear()
    worker_handle_mod._FTE_SEQUENCES.clear()
    worker_handle_mod._FTE_FRAGMENT_STATES.clear()
    worker_handle_mod._FTE_WORKER_HANDLES.clear()
    worker_handle_mod._FTE_WORKER_RESERVATION_GENERATIONS.clear()
    worker_handle_mod._FTE_PENDING_WORKER_RESERVATIONS.clear()
    worker_handle_mod._FTE_PARTITION_TASK_WAITERS.clear()
    worker_handle_mod._FTE_STAGE_SUBMISSION_PROBES.clear()
    worker_handle_mod._FTE_STAGE_SUBMISSION_BLOCKS.clear()
    worker_handle_mod._FTE_PARTITION_TASK_LEASES.clear()
    worker_handle_mod._FTE_RESULT_HANDLES_BY_QUERY.clear()
    worker_handle_mod._FTE_RETRY_DELAYS.clear()
    worker_handle_mod._FTE_SCHEDULERS.clear()
    worker_handle_mod._FTE_STATUS_WATCHERS.clear()
    worker_handle_mod._FTE_CLOSING_QUERIES.clear()
    worker_handle_mod._FTE_ACTIVE_OPERATIONS_BY_QUERY.clear()
    worker_handle_mod._FTE_ACTIVE_TEARDOWN_OPERATIONS_BY_QUERY.clear()
    monkeypatch.setenv("VANE_FTE_RETRY_INITIAL_DELAY_S", "0")
    monkeypatch.setattr(
        RayWorkerActorHandle,
        "_start_fte_attempt_status_watcher",
        lambda *_args, **_kwargs: None,
    )
    with worker_handle_mod._FRAGMENT_PLAN_REF_CACHE_LOCK:
        worker_handle_mod._FRAGMENT_PLAN_REF_CACHE.clear()
    monkeypatch.setattr(worker_handle_mod.ray, "get", lambda value, *_args, **_kwargs: value)
    monkeypatch.setattr(worker_handle_mod.ray, "put", lambda value, *_args, **_kwargs: value)
    monkeypatch.setattr(worker_handle_mod.ray, "wait", lambda refs, **_kwargs: (list(refs), []))
    monkeypatch.setattr(
        RayWorkerActorHandle,
        "_fte_task_handle_cls",
        staticmethod(lambda: _FakeFteTaskHandle),
    )

    original_submit_tasks = RayWorkerActorHandle.submit_tasks

    def _submit_tasks_with_registered_test_graph(handle, tasks):
        tasks = list(tasks)
        stages_by_query = {}
        for task in tasks:
            context = task.context()
            query_id, fragment_id = fragment_id_for_task(context, task.name())
            if ":node:" not in fragment_id:
                continue
            stages_by_query.setdefault(query_id, set()).add(fragment_id)
        for query_id, fragment_ids in stages_by_query.items():
            _register_test_query_graph(query_id, fragment_ids)
        return original_submit_tasks(handle, tasks)

    original_get_or_create = RayWorkerActorHandle._get_or_create_fte_fragment_execution

    def _get_or_create_with_registered_test_graph(handle, item, *args, **kwargs):
        query_id = str(item["query_id"])
        fragment_id = str(item["fragment_id"])
        if ":node:" in fragment_id:
            _register_test_query_graph(query_id, [fragment_id])
            stage_id = fte_stage_id_for_fragment(query_id, fragment_id)
            item = dict(item)
            item.setdefault("resource_query_id", query_id)
            item.setdefault("resource_stage_id", stage_id)
            item["context"] = {
                "resource_query_id": query_id,
                "resource_stage_id": stage_id,
                **dict(item.get("context") or {}),
            }
        return original_get_or_create(handle, item, *args, **kwargs)

    monkeypatch.setattr(RayWorkerActorHandle, "submit_tasks", _submit_tasks_with_registered_test_graph)
    monkeypatch.setattr(
        RayWorkerActorHandle,
        "_get_or_create_fte_fragment_execution",
        _get_or_create_with_registered_test_graph,
    )
    yield
    clear_query_resource_managers()


def test_fragment_plan_ref_cache_is_session_scoped(monkeypatch):
    session = "session-a"
    put_calls = []

    monkeypatch.setattr(
        worker_handle_mod,
        "_ray_fragment_plan_cache_session_key",
        lambda: session,
    )
    monkeypatch.setattr(
        worker_handle_mod.ray,
        "put",
        lambda value, *_args, **_kwargs: put_calls.append((session, value)) or f"ref:{session}:{value['plan']}",
    )

    ref_a = worker_handle_mod._fragment_plan_ref("query-cache", "query-cache:node:7", {"plan": "a"})
    ref_a_again = worker_handle_mod._fragment_plan_ref("query-cache", "query-cache:node:7", {"plan": "ignored"})
    session = "session-b"
    ref_b = worker_handle_mod._fragment_plan_ref("query-cache", "query-cache:node:7", {"plan": "b"})

    assert ref_a == "ref:session-a:a"
    assert ref_a_again == ref_a
    assert ref_b == "ref:session-b:b"
    assert put_calls == [
        ("session-a", {"plan": "a"}),
        ("session-b", {"plan": "b"}),
    ]


def test_fragment_plan_cache_drop_uses_exact_query_ownership(monkeypatch):
    monkeypatch.setattr(
        worker_handle_mod,
        "_ray_fragment_plan_cache_session_key",
        lambda: "query-isolation-session",
    )
    monkeypatch.setattr(
        worker_handle_mod.ray,
        "put",
        lambda value, *_args, **_kwargs: object(),
    )

    query_ref = worker_handle_mod._fragment_plan_ref("q", "q:node:1", {"plan": "q"})
    child_ref = worker_handle_mod._fragment_plan_ref("q:child", "q:child:node:1", {"plan": "child"})

    assert worker_handle_mod._drop_fragment_plan_refs_for_query("q") == 1
    assert query_ref not in worker_handle_mod._FRAGMENT_PLAN_REF_CACHE.values()
    assert child_ref in worker_handle_mod._FRAGMENT_PLAN_REF_CACHE.values()
    assert worker_handle_mod._drop_fragment_plan_refs_for_query("q:child") == 1


def _create_requests(actor):
    return [call[1] for call in actor.fte_calls if call[0] == "create"]


def test_submit_tasks_rejects_missing_query_id_before_registering_fragment():
    actor = _FakeActor()
    handle = RayWorkerActorHandle(actor, memory_capacity_bytes=1 << 60)
    task = _FakeTask(
        name="scan-task-missing-query",
        context={"node_id": "17"},
        inputs={"17": {"kind": "scan_task", "data": b"a"}},
        plan={"plan": "scan"},
    )

    with pytest.raises(ValueError, match="non-empty query_id"):
        handle.submit_tasks([task])

    assert actor.register_payloads == []
    assert actor.fte_calls == []
    assert task.plan_calls == 0


def test_submit_tasks_propagates_task_inputs_errors():
    actor = _FakeActor()
    handle = RayWorkerActorHandle(actor, memory_capacity_bytes=1 << 60)
    task = _InputsFailingTask(
        name="scan-task-inputs-fail",
        context={"query_id": "query-inputs-fail", "node_id": "17"},
        plan={"plan": "scan"},
    )

    with pytest.raises(RuntimeError, match="inputs exploded"):
        handle.submit_tasks([task])

    assert actor.register_payloads == []
    assert actor.fte_calls == []
    assert task.plan_calls == 0


def test_submit_tasks_rejects_task_without_inputs_method():
    actor = _FakeActor()
    handle = RayWorkerActorHandle(actor, memory_capacity_bytes=1 << 60)
    task = _MissingInputsTask(
        name="scan-task-no-inputs-method",
        context={"query_id": "query-no-inputs", "node_id": "17"},
        plan={"plan": "scan"},
    )

    with pytest.raises(TypeError, match="not callable"):
        handle.submit_tasks([task])

    assert actor.register_payloads == []
    assert actor.fte_calls == []
    assert task.plan_calls == 0


def test_submit_tasks_propagates_exchange_sink_instance_errors():
    actor = _FakeActor()
    handle = RayWorkerActorHandle(actor, memory_capacity_bytes=1 << 60)
    task = _ExchangeSinkInstanceFailingTask(
        name="scan-task-sink-fail",
        context={"query_id": "query-sink-fail", "node_id": "17"},
        plan={"plan": "scan"},
    )

    with pytest.raises(RuntimeError, match="exchange sink instance exploded"):
        handle.submit_tasks([task])

    assert actor.register_payloads == []
    assert actor.fte_calls == []
    assert task.plan_calls == 0


def test_submit_tasks_registers_fragment_and_creates_fte_task():
    actor = _FakeActor()
    handle = RayWorkerActorHandle(actor, memory_capacity_bytes=1 << 60)
    task = _FakeTask(
        name="scan-task-1",
        context={"query_id": "query-1", "node_id": "17"},
        plan={"plan": "scan"},
    )
    expected_fragment_id = fragment_id_for_task(task.context(), task.name())[1]

    handles = handle.submit_tasks([task])

    assert len(handles) == 1
    assert isinstance(handles[0], _FakeFteTaskHandle)
    assert actor.register_payloads == [
        [
            {
                "fragment_id": expected_fragment_id,
                "plan": {"plan": "scan"},
                "query_id": "query-1",
            }
        ]
    ]
    request = _create_requests(actor)[0]
    assert request["fragment_id"] == expected_fragment_id
    assert request["context"] == {
        "query_id": "query-1",
        "node_id": "17",
        "resource_query_id": "query-1",
        "resource_stage_id": "stage:query-1:node:17:fte",
    }
    assert request["worker_runtime"] == "fte"
    assert request["fragment_plan"] is None
    assert request["query_task_lease"]["stage_id"] == "stage:query-1:node:17:fte"
    assert request["query_task_lease"]["attempt_id"] == str(handles[0].task_id)
    assert request["query_task_lease"]["resources"]["heap_bytes"] == 10
    assert "duckdb_memory_bytes" not in request["query_task_lease"]
    assert request["memory_requirement_bytes"] == 10

    assert handle.submit_tasks([task]) == []
    assert len(actor.register_payloads) == 1
    assert len(_create_requests(actor)) == 1
    assert task.plan_calls == 1


def test_submit_tasks_rejects_fragment_without_pre_registered_physical_node_id():
    actor = _FakeActor()
    handle = RayWorkerActorHandle(actor, memory_capacity_bytes=1 << 60)
    task = _FakeTask(
        name="aggregate-task-1",
        context={"query_id": "query-2"},
        plan={"plan": "aggregate"},
    )

    with pytest.raises(ValueError, match="requires resource_query_id and resource_stage_id"):
        handle.submit_tasks([task])

    assert _create_requests(actor) == []


def test_submit_tasks_creates_fte_tasks_for_distinct_fragments():
    actor = _FakeActor()
    handle = RayWorkerActorHandle(actor, memory_capacity_bytes=1 << 60)
    tasks = [
        _FakeTask(name="scan-task-a", context={"query_id": "query-3", "node_id": "3"}),
        _FakeTask(name="scan-task-b", context={"query_id": "query-3", "node_id": "5"}),
    ]
    expected_ids = [fragment_id_for_task(task.context(), task.name())[1] for task in tasks]

    handles = handle.submit_tasks(tasks)

    assert len(handles) == 2
    assert [request["fragment_id"] for request in _create_requests(actor)] == expected_ids
    assert actor.register_payloads == [
        [
            {
                "fragment_id": expected_ids[0],
                "plan": {"plan": "scan-task-a"},
                "query_id": "query-3",
            },
            {
                "fragment_id": expected_ids[1],
                "plan": {"plan": "scan-task-b"},
                "query_id": "query-3",
            },
        ]
    ]
    assert actor.fragment_calls == []


def test_submit_tasks_coalesces_same_fragment_scan_splits_in_fte_stage():
    actor = _FakeActor()
    handle = RayWorkerActorHandle(actor, memory_capacity_bytes=1 << 60)
    task0 = _FakeTask(
        name="scan-task-0",
        context={"query_id": "query-merge", "node_id": "7"},
        inputs={"7": {"kind": "scan_task", "data": b"a"}},
        plan={"plan": "scan-template"},
    )
    task1 = _FakeTask(
        name="scan-task-1",
        context={"query_id": "query-merge", "node_id": "7"},
        inputs={"7": {"kind": "scan_task", "data": b"b"}},
        plan={"plan": "scan-template"},
    )
    expected_fragment_id = fragment_id_for_task(task0.context(), task0.name())[1]

    handles = handle.submit_tasks([task0, task1])

    assert len(handles) == 1
    request = _create_requests(actor)[0]
    assert request["fragment_id"] == expected_fragment_id
    assert "scan_task:7" not in request["context"]
    assert "scan_task_nodes" not in request["context"]
    assert request["dynamic_scan_source_node_ids"] == ["7"]
    assert [split["data"] for split in request["initial_splits"]["7"]] == [b"a", b"b"]
    assert actor.register_payloads == [
        [
            {
                "fragment_id": expected_fragment_id,
                "plan": {"plan": "scan-template"},
                "query_id": "query-merge",
            }
        ]
    ]
    assert task0.plan_calls == 1
    assert task1.plan_calls == 0


def test_submit_tasks_allows_copy_tasks_with_attempt_aware_final_writes():
    actor = _FakeActor()
    handle = RayWorkerActorHandle(actor, memory_capacity_bytes=1 << 60)
    task = _FakeTask(
        name="copy-task",
        context={
            "query_id": "query-copy",
            "node_id": "42",
            "copy_output_base": "",
            "copy_output_run_id": "run-copy",
            "copy_output_remote_base": "/tmp/task.parquet",
        },
        plan={"plan": "copy-template"},
    )

    handles = handle.submit_tasks([task])

    assert len(handles) == 1
    request = _create_requests(actor)[0]
    assert request["context"]["copy_output_remote_base"] == "/tmp/task.parquet"
    assert actor.fragment_calls == []
    assert task.plan_calls == 1


def test_submit_tasks_rejects_variant_fragment_ids_outside_physical_stage_identity():
    actor = _FakeActor()
    handle = RayWorkerActorHandle(actor, memory_capacity_bytes=1 << 60)
    task0 = _FakeTask(
        name="exchange-task-0",
        context={
            "query_id": "query-exchange",
            "node_id": "42",
            "fragment_id": "query-exchange:node:42:variant:a",
        },
        plan={"plan": "exchange-a"},
    )
    task1 = _FakeTask(
        name="exchange-task-1",
        context={
            "query_id": "query-exchange",
            "node_id": "42",
            "fragment_id": "query-exchange:node:42:variant:b",
        },
        plan={"plan": "exchange-b"},
    )
    with pytest.raises(ValueError, match="invalid FTE fragment_id"):
        handle.submit_tasks([task0, task1])

    assert _create_requests(actor) == []


def test_fte_worker_actor_handle_wraps_control_rpcs():
    actor = _FakeActor()
    handle = RayWorkerActorHandle(actor, memory_capacity_bytes=1 << 60)
    handle._registered_fragment_ids.update({"q:node:a", "other:node:b"})
    handle._fragment_query_ids.update({"q:node:a": "q", "other:node:b": "other"})
    task_id = {"query_id": "q", "fragment_execution_id": 0, "partition_id": 1, "attempt_id": 0}
    request = {"task_id": task_id, "fragment_id": "q:node:a"}

    assert handle.fte_create_task(request)["state"] == "RUNNING"
    assert handle.fte_add_splits(task_id, "7", [{"sequence_id": 1}])["version"] == 2
    assert handle.fte_no_more_splits(task_id, "7")["version"] == 3
    assert handle.fte_update_task(task_id, {"output_buffers": {"version": 1}})["version"] == 4
    assert handle.fte_get_task_status(task_id)["state"] == "FINISHED"
    assert handle.fte_wait_task_status(task_id, 3, 0.01)["state"] == "FINISHED"
    assert handle.fte_wait_split_queue_has_space(task_id, "7", 4, 0.01)["has_space"] is True
    assert handle.fte_get_task_info(task_id)["status"]["state"] == "FINISHED"
    ack_ref = handle.enqueue_fte_ack_task_result(task_id)
    release_ref = handle.enqueue_fte_release_task_result(task_id)
    assert isinstance(ack_ref, _ImmediateObjectRef)
    assert isinstance(release_ref, _ImmediateObjectRef)
    assert handle.fte_cancel_task(task_id)["state"] == "CANCELED"
    assert handle.fte_drop_query("q") == {
        "tasks_removed": 1,
        "tasks_canceled": 0,
        "fragments_removed": 2,
    }

    assert [call[0] for call in actor.fte_calls] == [
        "create",
        "add_splits",
        "no_more_splits",
        "update_task",
        "get_status",
        "wait_status",
        "wait_split_queue",
        "get_info",
        "ack",
        "release",
        "cancel",
        "drop_query",
    ]
    assert handle._registered_fragment_ids == {"other:node:b"}


def test_fte_status_wait_preserves_query_deadline_failure(monkeypatch):
    monkeypatch.setenv("VANE_QUERY_DEADLINE_EPOCH_S", "1")
    actor = _FakeActor()
    handle = RayWorkerActorHandle(actor, memory_capacity_bytes=1 << 60)
    task_id = {
        "query_id": "q",
        "fragment_execution_id": 0,
        "partition_id": 1,
        "attempt_id": 0,
    }

    with pytest.raises(QueryDeadlineExceeded, match="query deadline expired"):
        handle.fte_wait_task_status(task_id, 3, 0.01)


def test_fte_worker_actor_handle_chains_async_control_updates_by_task():
    class _RecordingActor:
        def __init__(self):
            self.calls = []
            self.fte_add_splits = _FakeRemoteMethod(self._fte_add_splits)
            self.fte_no_more_splits = _FakeRemoteMethod(self._fte_no_more_splits)
            self.fte_update_task = _FakeRemoteMethod(self._fte_update_task)

        def _fte_add_splits(
            self,
            task_id,
            source_node_id,
            splits,
            dependency=None,
        ):
            self.calls.append(("add_splits", task_id, source_node_id, splits, dependency))
            return {"state": "RUNNING", "ref": "add-ref"}

        def _fte_no_more_splits(self, task_id, source_node_id, dependency=None):
            self.calls.append(("no_more_splits", task_id, source_node_id, dependency))
            return {"state": "RUNNING", "ref": "no-more-ref"}

        def _fte_update_task(self, task_id, update, dependency=None):
            self.calls.append(("update_task", task_id, update, dependency))
            return {"state": "RUNNING", "ref": "update-ref"}

    actor = _RecordingActor()
    handle = RayWorkerActorHandle(actor, memory_capacity_bytes=1 << 60)
    task_id = {"query_id": "q", "fragment_execution_id": 0, "partition_id": 1, "attempt_id": 0}
    other_task_id = {"query_id": "q", "fragment_execution_id": 0, "partition_id": 2, "attempt_id": 0}

    assert handle.enqueue_fte_add_splits(task_id, "7", [{"sequence_id": 1}])["ref"] == "add-ref"
    assert handle.enqueue_fte_no_more_splits(task_id, "7")["ref"] == "no-more-ref"
    assert handle.enqueue_fte_update_task(task_id, {"output_buffers": {"version": 1}})["ref"] == "update-ref"
    assert handle.enqueue_fte_no_more_splits(other_task_id, "7")["ref"] == "no-more-ref"

    assert actor.calls == [
        ("add_splits", task_id, "7", [{"sequence_id": 1}], None),
        ("no_more_splits", task_id, "7", {"state": "RUNNING", "ref": "add-ref"}),
        (
            "update_task",
            task_id,
            {"output_buffers": {"version": 1}},
            {"state": "RUNNING", "ref": "no-more-ref"},
        ),
        ("no_more_splits", other_task_id, "7", None),
    ]


def test_fte_worker_actor_handle_defers_ordered_result_controls_until_query_barrier(monkeypatch):
    class _Ref:
        def __init__(self, name):
            self.name = name

    class _DeferredRemoteMethod:
        def __init__(self, name, calls):
            self._name = name
            self._calls = calls

        def remote(self, *args):
            ref = _Ref(f"{self._name}-{len(self._calls)}")
            self._calls.append((self._name, args, ref))
            return ref

    class _DeferredActor:
        def __init__(self):
            self.calls = []
            self.fte_ack_task_result = _DeferredRemoteMethod("ack", self.calls)
            self.fte_release_task_result = _DeferredRemoteMethod("release", self.calls)

    resolved = []

    def _resolve(refs, *, timeout=None, honor_query_deadline=True):
        assert honor_query_deadline is False
        resolved.append((list(refs), timeout))
        return [
            {
                "state": "FINISHED",
                "task_id": task_id,
                "_fte_control_operation": "fte_release_task_result",
                "_fte_control_applied": True,
            }
            for _ in refs
        ]

    monkeypatch.setattr(task_control_mod, "resolve_object_refs_blocking", _resolve)
    actor = _DeferredActor()
    handle = RayWorkerActorHandle(actor, memory_capacity_bytes=1 << 60)
    task_id = {"query_id": "q", "fragment_execution_id": 0, "partition_id": 1, "attempt_id": 0}

    ack_ref = handle.enqueue_fte_ack_task_result(task_id)
    release_ref = handle.enqueue_fte_release_task_result(task_id)

    assert resolved == []
    assert actor.calls == [
        ("ack", (task_id,), ack_ref),
        ("release", (task_id, ack_ref), release_ref),
    ]

    handle.close_and_flush_fte_controls("q")
    handle.close_and_flush_fte_controls("q")

    assert resolved == [([release_ref], 30.0)]
    with pytest.raises(RuntimeError, match="control admission is closed"):
        handle.enqueue_fte_ack_task_result(task_id)
    assert handle.enqueue_fte_release_task_result(task_id) is None
    assert len(actor.calls) == 2


def test_fte_control_barrier_rejects_contradictory_status_identities(monkeypatch):
    class _Ref:
        pass

    class _DeferredRemoteMethod:
        def __init__(self):
            self.ref = _Ref()

        def remote(self, *_args):
            return self.ref

    class _DeferredActor:
        def __init__(self):
            self.fte_ack_task_result = _DeferredRemoteMethod()

    task_id = {
        "query_id": "q-control-identity",
        "fragment_execution_id": 0,
        "partition_id": 1,
        "attempt_id": 0,
    }
    mismatched_task_id = {
        **task_id,
        "partition_id": 2,
    }

    def _resolve(refs, *, timeout=None, honor_query_deadline=True):
        assert honor_query_deadline is False
        assert refs == [actor.fte_ack_task_result.ref]
        assert timeout == 30.0
        return [
            {
                "state": "FINISHED",
                "task_id": task_id,
                "task_id_string": str(FteTaskAttemptId.coerce(mismatched_task_id)),
                "_fte_control_operation": "fte_ack_task_result",
                "_fte_control_applied": True,
            }
        ]

    monkeypatch.setattr(task_control_mod, "resolve_object_refs_blocking", _resolve)
    actor = _DeferredActor()
    handle = RayWorkerActorHandle(actor, memory_capacity_bytes=1 << 60)
    handle.enqueue_fte_ack_task_result(task_id)

    with pytest.raises(RuntimeError, match="status identity mismatch"):
        handle.close_and_flush_fte_controls("q-control-identity")


def test_worker_control_status_rejects_contradictory_status_identities():
    task_id = {
        "query_id": "q-worker-control-identity",
        "fragment_execution_id": 0,
        "partition_id": 1,
        "attempt_id": 0,
    }
    mismatched_task_id = {**task_id, "partition_id": 2}

    with pytest.raises(RuntimeError, match="mismatched task identity"):
        worker_mod._fte_applied_control_status(
            "fte_ack_task_result",
            task_id,
            {
                "state": "FINISHED",
                "task_id": task_id,
                "task_id_string": str(FteTaskAttemptId.coerce(mismatched_task_id)),
            },
        )


def test_create_task_timeout_retains_remote_mutation_ownership(monkeypatch):
    class _DeferredFuture:
        def __init__(self):
            self.callbacks = []
            self.done = False
            self.value = None

        def add_done_callback(self, callback):
            if self.done:
                callback(self)
            else:
                self.callbacks.append(callback)

        def result(self, timeout=None):
            if not self.done:
                raise TimeoutError("create is still pending")
            return self.value

        def complete(self, value):
            self.value = value
            self.done = True
            callbacks = list(self.callbacks)
            self.callbacks.clear()
            for callback in callbacks:
                callback(self)

    class _DeferredRef:
        def __init__(self, future):
            self._future = future

        def future(self):
            return self._future

    class _DeferredCreate:
        def __init__(self, ref):
            self.ref = ref

        def remote(self, _request):
            return self.ref

    query_id = "query-create-timeout-fence"
    task_id = {
        "query_id": query_id,
        "fragment_execution_id": 0,
        "partition_id": 0,
        "attempt_id": 0,
    }
    future = _DeferredFuture()
    actor = _FakeActor()
    actor.fte_create_task = _DeferredCreate(_DeferredRef(future))
    handle = RayWorkerActorHandle(actor, memory_capacity_bytes=1 << 60)
    monkeypatch.setattr(
        task_control_mod,
        "resolve_object_refs_blocking",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(TimeoutError("planned create timeout")),
    )

    with pytest.raises(TimeoutError, match="did not complete"):
        handle.fte_create_task({"task_id": task_id, "fragment_id": f"{query_id}:node:1"})

    assert worker_handle_mod._FTE_ACTIVE_OPERATIONS_BY_QUERY[query_id] == 1
    assert future.done is False
    close_done = threading.Event()

    def close_registry():
        worker_handle_mod.close_fte_registry_for_query(query_id)
        worker_handle_mod.quiesce_fte_registry_for_query(query_id)
        close_done.set()

    close_thread = threading.Thread(target=close_registry)
    close_thread.start()
    time.sleep(0.05)
    assert close_done.is_set() is False

    future.complete(
        {
            "state": "RUNNING",
            "task_id": task_id,
            "_fte_control_operation": "fte_create_task",
            "_fte_control_applied": True,
        }
    )
    close_thread.join(2.0)
    assert close_done.is_set() is True
    assert query_id not in worker_handle_mod._FTE_ACTIVE_OPERATIONS_BY_QUERY
    assert handle._has_fte_control_state_for_query(query_id) is False
    worker_handle_mod.open_fte_registry_for_query(query_id)


def test_remote_drop_timeout_retains_fence_and_local_generation(monkeypatch):
    class _DeferredFuture:
        def __init__(self):
            self.callbacks = []
            self.done = False
            self.value = None

        def add_done_callback(self, callback):
            if self.done:
                callback(self)
            else:
                self.callbacks.append(callback)

        def result(self, timeout=None):
            if not self.done:
                raise TimeoutError("drop is still pending")
            return self.value

        def complete(self, value):
            self.value = value
            self.done = True
            callbacks = list(self.callbacks)
            self.callbacks.clear()
            for callback in callbacks:
                callback(self)

    class _DeferredRef:
        def __init__(self, future):
            self._future = future

        def future(self):
            return self._future

    class _DeferredDrop:
        def __init__(self, ref):
            self.ref = ref

        def remote(self, _query_id):
            return self.ref

    query_id = "query-remote-drop-timeout-fence"
    future = _DeferredFuture()
    actor = _FakeActor()
    actor.fte_drop_query = _DeferredDrop(_DeferredRef(future))
    handle = RayWorkerActorHandle(actor, memory_capacity_bytes=1 << 60)
    fragment_id = f"{query_id}:node:1"
    with handle._fragment_registration_lock:
        handle._registered_fragment_ids.add(fragment_id)
        handle._fragment_query_ids[fragment_id] = query_id
    with worker_handle_mod._FRAGMENT_PLAN_REF_CACHE_LOCK:
        worker_handle_mod._FRAGMENT_PLAN_REF_CACHE[("drop-timeout-test-session", query_id, fragment_id)] = object()
    monkeypatch.setattr(
        task_control_mod,
        "resolve_object_refs_blocking",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(TimeoutError("planned remote drop timeout")),
    )

    with pytest.raises(RuntimeError, match="did not complete"):
        handle.fte_drop_query(query_id)

    assert worker_handle_mod._FTE_ACTIVE_TEARDOWN_OPERATIONS_BY_QUERY[query_id] == 1
    assert fragment_id in handle._registered_fragment_ids
    with pytest.raises(RuntimeError, match="active_teardown_operations"):
        worker_handle_mod.open_fte_registry_for_query(query_id)

    future.complete({"tasks_removed": 0, "tasks_canceled": 0, "fragments_removed": 0})
    assert query_id not in worker_handle_mod._FTE_ACTIVE_TEARDOWN_OPERATIONS_BY_QUERY
    with pytest.raises(RuntimeError, match="fragment_plan_refs"):
        worker_handle_mod.open_fte_registry_for_query(query_id)
    monkeypatch.setattr(
        task_control_mod,
        "resolve_object_refs_blocking",
        lambda ref, **_kwargs: ref.future().result(),
    )
    assert handle.fte_drop_query(query_id) == {
        "tasks_removed": 0,
        "tasks_canceled": 0,
        "fragments_removed": 0,
    }
    worker_handle_mod.open_fte_registry_for_query(query_id)


def test_pending_teardown_on_one_worker_does_not_block_drop_fanout(monkeypatch):
    class _DeferredFuture:
        def __init__(self):
            self.callbacks = []
            self.done = False
            self.value = None

        def add_done_callback(self, callback):
            if self.done:
                callback(self)
            else:
                self.callbacks.append(callback)

        def result(self, timeout=None):
            if not self.done:
                raise TimeoutError("drop is pending")
            return self.value

        def complete(self, value):
            self.value = value
            self.done = True
            callbacks = list(self.callbacks)
            self.callbacks.clear()
            for callback in callbacks:
                callback(self)

    class _Ref:
        def __init__(self, future):
            self._future = future

        def future(self):
            return self._future

    class _DropMethod:
        def __init__(self, ref):
            self.ref = ref
            self.calls = []

        def remote(self, query_id):
            self.calls.append(query_id)
            return self.ref

    query_id = "query-drop-fanout-pending-worker"
    pending_future = _DeferredFuture()
    pending_ref = _Ref(pending_future)
    actor1 = _FakeActor()
    actor1.fte_drop_query = _DropMethod(pending_ref)
    actor2 = _FakeActor()
    handle1 = RayWorkerActorHandle(
        actor1,
        memory_capacity_bytes=1 << 60,
        worker_id="drop-fanout-worker-1",
    )
    handle2 = RayWorkerActorHandle(
        actor2,
        memory_capacity_bytes=1 << 60,
        worker_id="drop-fanout-worker-2",
    )

    def resolve(ref, **_kwargs):
        if ref is pending_ref and not pending_future.done:
            raise TimeoutError("planned first-worker drop timeout")
        return ref.future().result()

    monkeypatch.setattr(task_control_mod, "resolve_object_refs_blocking", resolve)

    with pytest.raises(RuntimeError, match="did not complete"):
        handle1.fte_drop_query(query_id)

    assert handle2.fte_drop_query(query_id) == {
        "tasks_removed": 1,
        "tasks_canceled": 0,
        "fragments_removed": 2,
    }
    assert actor1.fte_drop_query.calls == [query_id]
    assert [call for call in actor2.fte_calls if call[0] == "drop_query"] == [("drop_query", query_id)]

    pending_future.complete({"tasks_removed": 0, "tasks_canceled": 0, "fragments_removed": 0})
    assert handle1.fte_drop_query(query_id) == {
        "tasks_removed": 0,
        "tasks_canceled": 0,
        "fragments_removed": 0,
    }
    worker_handle_mod.open_fte_registry_for_query(query_id)


def test_pending_control_barrier_does_not_submit_remote_drop(monkeypatch):
    class _Future:
        def __init__(self):
            self.done = False
            self.value = None

        def result(self, timeout=None):
            if not self.done:
                raise TimeoutError("control is pending")
            return self.value

        def complete(self, value):
            self.value = value
            self.done = True

    class _Ref:
        def __init__(self, future):
            self._future = future

        def future(self):
            return self._future

    class _AckMethod:
        def __init__(self, ref):
            self.ref = ref

        def remote(self, *_args):
            return self.ref

    query_id = "query-pending-control-before-drop"
    task_id = {
        "query_id": query_id,
        "fragment_execution_id": 0,
        "partition_id": 0,
        "attempt_id": 0,
    }
    future = _Future()
    actor = _FakeActor()
    actor.fte_ack_task_result = _AckMethod(_Ref(future))
    handle = RayWorkerActorHandle(actor, memory_capacity_bytes=1 << 60)
    handle.enqueue_fte_ack_task_result(task_id)

    def pending_resolve(refs, **_kwargs):
        if isinstance(refs, list):
            raise TimeoutError("planned pending control barrier")
        return refs.future().result()

    monkeypatch.setattr(
        task_control_mod,
        "resolve_object_refs_blocking",
        pending_resolve,
    )

    with pytest.raises(
        task_control_mod.FteControlBarrierPendingError,
        match="retained pending control ownership",
    ):
        handle.fte_drop_query(query_id)

    assert [call for call in actor.fte_calls if call[0] == "drop_query"] == []
    assert handle._has_fte_control_state_for_query(query_id) is True
    assert handle._has_fte_teardown_state_for_query(query_id) is True

    future.complete(
        {
            "state": "FINISHED",
            "task_id": task_id,
            "_fte_control_operation": "fte_ack_task_result",
            "_fte_control_applied": True,
        }
    )

    def completed_resolve(refs, **_kwargs):
        if isinstance(refs, list):
            return [ref.future().result() for ref in refs]
        return refs.future().result()

    monkeypatch.setattr(
        task_control_mod,
        "resolve_object_refs_blocking",
        completed_resolve,
    )
    assert handle.fte_drop_query(query_id) == {
        "tasks_removed": 1,
        "tasks_canceled": 0,
        "fragments_removed": 2,
    }
    worker_handle_mod.open_fte_registry_for_query(query_id)


def test_terminal_failed_control_allows_drop_and_clears_ownership(monkeypatch):
    class _FailedFuture:
        def result(self, timeout=None):
            raise RuntimeError("planned terminal control failure")

    class _Ref:
        def future(self):
            return _FailedFuture()

    class _AckMethod:
        def remote(self, *_args):
            return _Ref()

    query_id = "query-terminal-control-before-drop"
    task_id = {
        "query_id": query_id,
        "fragment_execution_id": 0,
        "partition_id": 0,
        "attempt_id": 0,
    }
    actor = _FakeActor()
    actor.fte_ack_task_result = _AckMethod()
    handle = RayWorkerActorHandle(actor, memory_capacity_bytes=1 << 60)
    handle.enqueue_fte_ack_task_result(task_id)

    def resolve(refs, **_kwargs):
        if isinstance(refs, list):
            raise RuntimeError("planned terminal control failure")
        return refs.future().result()

    original_drop_ref = actor.fte_drop_query

    def resolve_with_drop(refs, **kwargs):
        if isinstance(refs, list):
            return resolve(refs, **kwargs)
        return refs.future().result()

    monkeypatch.setattr(
        task_control_mod,
        "resolve_object_refs_blocking",
        resolve_with_drop,
    )

    with pytest.raises(RuntimeError, match="planned terminal control failure"):
        handle.fte_drop_query(query_id)

    assert original_drop_ref is actor.fte_drop_query
    assert [call for call in actor.fte_calls if call[0] == "drop_query"] == [("drop_query", query_id)]
    assert handle._has_fte_control_state_for_query(query_id) is False
    assert handle._has_fte_teardown_state_for_query(query_id) is False
    worker_handle_mod.open_fte_registry_for_query(query_id)


def test_control_barrier_ignores_stale_bulk_timeout_after_terminal_probe(monkeypatch):
    class _CompletedFuture:
        def __init__(self, value):
            self.value = value

        def result(self, timeout=None):
            return self.value

    class _Ref:
        def __init__(self, value):
            self._future = _CompletedFuture(value)

        def future(self):
            return self._future

    query_id = "query-control-completed-after-bulk-timeout"
    task_id = {
        "query_id": query_id,
        "fragment_execution_id": 0,
        "partition_id": 0,
        "attempt_id": 0,
    }
    expected_status = {
        "state": "FINISHED",
        "task_id": task_id,
        "_fte_control_operation": "fte_ack_task_result",
        "_fte_control_applied": True,
    }

    class _AckMethod:
        def remote(self, *_args):
            return _Ref(expected_status)

    actor = _FakeActor()
    actor.fte_ack_task_result = _AckMethod()
    handle = RayWorkerActorHandle(actor, memory_capacity_bytes=1 << 60)
    handle.enqueue_fte_ack_task_result(task_id)

    def bulk_timeout(refs, **_kwargs):
        if isinstance(refs, list):
            raise TimeoutError("bulk deadline raced terminal completion")
        return refs.future().result()

    monkeypatch.setattr(
        task_control_mod,
        "resolve_object_refs_blocking",
        bulk_timeout,
    )

    assert handle.close_and_flush_fte_controls(query_id) == [expected_status]
    assert handle._has_fte_control_state_for_query(query_id) is False
    worker_handle_mod.open_fte_registry_for_query(query_id)


def test_done_control_future_with_timeout_error_is_terminal(monkeypatch):
    class _TerminalTimeoutFuture:
        def done(self):
            return True

        def result(self, timeout=None):
            raise TimeoutError("remote control terminated with timeout")

    class _Ref:
        def future(self):
            return _TerminalTimeoutFuture()

    class _AckMethod:
        def remote(self, *_args):
            return _Ref()

    query_id = "query-terminal-timeout-control"
    task_id = {
        "query_id": query_id,
        "fragment_execution_id": 0,
        "partition_id": 0,
        "attempt_id": 0,
    }
    actor = _FakeActor()
    actor.fte_ack_task_result = _AckMethod()
    handle = RayWorkerActorHandle(actor, memory_capacity_bytes=1 << 60)
    handle.enqueue_fte_ack_task_result(task_id)

    def resolve(refs, **_kwargs):
        if isinstance(refs, list):
            raise TimeoutError("bulk barrier observed terminal timeout")
        return refs.future().result()

    monkeypatch.setattr(task_control_mod, "resolve_object_refs_blocking", resolve)

    with pytest.raises(RuntimeError, match="remote control terminated with timeout"):
        handle.fte_drop_query(query_id)

    assert [call for call in actor.fte_calls if call[0] == "drop_query"] == [("drop_query", query_id)]
    assert handle._has_fte_control_state_for_query(query_id) is False
    assert handle._has_fte_teardown_state_for_query(query_id) is False
    worker_handle_mod.open_fte_registry_for_query(query_id)


def test_control_future_completed_during_zero_timeout_probe_is_reloaded(monkeypatch):
    query_id = "query-control-completed-during-probe"
    task_id = {
        "query_id": query_id,
        "fragment_execution_id": 0,
        "partition_id": 0,
        "attempt_id": 0,
    }
    expected_status = {
        "state": "FINISHED",
        "task_id": task_id,
        "_fte_control_operation": "fte_ack_task_result",
        "_fte_control_applied": True,
    }

    class _RacingFuture:
        def __init__(self):
            self.calls = 0

        def done(self):
            return True

        def result(self, timeout=None):
            self.calls += 1
            if self.calls == 1:
                raise TimeoutError("zero-time probe raced completion")
            return expected_status

    future = _RacingFuture()

    class _Ref:
        def future(self):
            return future

    class _AckMethod:
        def remote(self, *_args):
            return _Ref()

    actor = _FakeActor()
    actor.fte_ack_task_result = _AckMethod()
    handle = RayWorkerActorHandle(actor, memory_capacity_bytes=1 << 60)
    handle.enqueue_fte_ack_task_result(task_id)

    monkeypatch.setattr(
        task_control_mod,
        "resolve_object_refs_blocking",
        lambda refs, **_kwargs: (
            (_ for _ in ()).throw(TimeoutError("bulk barrier timed out"))
            if isinstance(refs, list)
            else refs.future().result()
        ),
    )

    assert handle.close_and_flush_fte_controls(query_id) == [expected_status]
    assert future.calls == 2
    assert handle._has_fte_control_state_for_query(query_id) is False
    worker_handle_mod.open_fte_registry_for_query(query_id)


def test_fte_control_close_fences_concurrent_late_admission(monkeypatch):
    monkeypatch.setenv("VANE_FTE_CONTROL_RPC_MAX_ATTEMPTS", "1")

    class _Ref:
        pass

    class _DeferredRemoteMethod:
        def __init__(self):
            self.ref = _Ref()

        def remote(self, *_args):
            return self.ref

    class _DeferredActor:
        def __init__(self):
            self.fte_ack_task_result = _DeferredRemoteMethod()

    task_id = {
        "query_id": "q-close-race",
        "fragment_execution_id": 0,
        "partition_id": 1,
        "attempt_id": 0,
    }
    barrier_started = threading.Event()
    barrier_release = threading.Event()

    def _resolve(refs, *, timeout=None, honor_query_deadline=True):
        assert honor_query_deadline is False
        assert timeout == 30.0
        assert refs == [actor.fte_ack_task_result.ref]
        barrier_started.set()
        assert barrier_release.wait(timeout=1.0)
        return [
            {
                "state": "FINISHED",
                "task_id": task_id,
                "_fte_control_operation": "fte_ack_task_result",
                "_fte_control_applied": True,
            }
        ]

    monkeypatch.setattr(task_control_mod, "resolve_object_refs_blocking", _resolve)
    actor = _DeferredActor()
    handle = RayWorkerActorHandle(actor, memory_capacity_bytes=1 << 60)
    handle.enqueue_fte_ack_task_result(task_id)

    barrier_result = []
    barrier_error = []

    def _close():
        try:
            barrier_result.extend(handle.close_and_flush_fte_controls("q-close-race"))
        except BaseException as exc:  # pragma: no cover - asserted below
            barrier_error.append(exc)

    close_thread = threading.Thread(target=_close)
    close_thread.start()
    assert barrier_started.wait(timeout=1.0)

    # The close flag and tail snapshot are one critical section. Once the
    # barrier starts resolving its stable snapshot, no later mutation can enter.
    with pytest.raises(RuntimeError, match="control admission is closed"):
        handle.enqueue_fte_ack_task_result(task_id)

    barrier_release.set()
    close_thread.join(timeout=1.0)
    assert close_thread.is_alive() is False
    assert barrier_error == []
    assert len(barrier_result) == 1


def test_teardown_controls_ignore_expired_query_deadline(monkeypatch):
    query_id = "query-teardown-expired-deadline"
    task_id = {
        "query_id": query_id,
        "fragment_execution_id": 0,
        "partition_id": 0,
        "attempt_id": 0,
    }
    actor = _FakeActor()
    handle = RayWorkerActorHandle(actor, memory_capacity_bytes=1 << 60)
    handle.enqueue_fte_ack_task_result(task_id)
    monkeypatch.setenv("VANE_QUERY_DEADLINE_EPOCH_S", "1")

    assert handle.fte_drop_query(query_id) == {
        "tasks_removed": 1,
        "tasks_canceled": 0,
        "fragments_removed": 2,
    }

    assert handle._has_fte_control_state_for_query(query_id) is False
    assert handle._has_fte_teardown_state_for_query(query_id) is False
    worker_handle_mod.open_fte_registry_for_query(query_id)


def test_fte_worker_actor_handle_async_control_requires_dict_response(monkeypatch):
    actor = _FakeActor()
    handle = RayWorkerActorHandle(actor, memory_capacity_bytes=1 << 60)
    task_id = {"query_id": "q", "fragment_execution_id": 0, "partition_id": 1, "attempt_id": 0}
    monkeypatch.setattr(handle, "_enqueue_ordered_fte_control_rpc", lambda *_args, **_kwargs: None)

    with pytest.raises(TypeError, match="worker actor fte_add_splits must return a dict"):
        handle.enqueue_fte_add_splits(task_id, "7", [{"sequence_id": 1}])
    with pytest.raises(TypeError, match="worker actor fte_no_more_splits must return a dict"):
        handle.enqueue_fte_no_more_splits(task_id, "7")
    with pytest.raises(TypeError, match="worker actor fte_update_task must return a dict"):
        handle.enqueue_fte_update_task(task_id, {"output_buffers": {"version": 1}})


def test_fte_worker_actor_handle_async_control_waits_for_remote_failure(monkeypatch):
    monkeypatch.setenv("VANE_FTE_CONTROL_RPC_MAX_ATTEMPTS", "1")

    class _FailingActor:
        def __init__(self):
            self.calls = []
            self.fte_no_more_splits = _FakeRemoteMethod(self._fte_no_more_splits)

        def _fte_no_more_splits(self, task_id, source_node_id, dependency=None):
            self.calls.append(("no_more_splits", task_id, source_node_id, dependency))
            return "failing-ref"

    def ray_get(value, *_args, **_kwargs):
        if value == "failing-ref":
            raise RuntimeError("remote control failed")
        return value

    monkeypatch.setattr(worker_handle_mod.ray, "get", ray_get)
    actor = _FailingActor()
    handle = RayWorkerActorHandle(actor, memory_capacity_bytes=1 << 60)
    task_id = {"query_id": "q", "fragment_execution_id": 0, "partition_id": 1, "attempt_id": 0}

    with pytest.raises(RuntimeError, match="remote control failed"):
        handle.enqueue_fte_no_more_splits(task_id, "7")

    assert actor.calls == [("no_more_splits", task_id, "7", None)]


def test_fte_worker_actor_handle_async_control_does_not_retry_remote_failure(monkeypatch):
    monkeypatch.setenv("VANE_FTE_CONTROL_RPC_MAX_ATTEMPTS", "2")
    monkeypatch.setenv("VANE_FTE_CONTROL_RPC_INITIAL_BACKOFF_S", "0")

    class _FailingRemoteActor:
        def __init__(self):
            self.calls = []
            self.fte_no_more_splits = _FakeRemoteMethod(self._fte_no_more_splits)

        def _fte_no_more_splits(self, task_id, source_node_id, dependency=None):
            self.calls.append(("no_more_splits", task_id, source_node_id, dependency))
            return f"ref-{len(self.calls)}"

    def ray_get(_value, *_args, **_kwargs):
        raise RuntimeError("remote control failed")

    monkeypatch.setattr(worker_handle_mod.ray, "get", ray_get)
    actor = _FailingRemoteActor()
    handle = RayWorkerActorHandle(actor, memory_capacity_bytes=1 << 60)
    task_id = {"query_id": "q", "fragment_execution_id": 0, "partition_id": 1, "attempt_id": 0}

    with pytest.raises(RuntimeError, match="remote control failed"):
        handle.enqueue_fte_no_more_splits(task_id, "7")

    assert actor.calls == [("no_more_splits", task_id, "7", None)]


def test_fte_worker_actor_handle_async_control_retries_submission_failure(monkeypatch):
    monkeypatch.setenv("VANE_FTE_CONTROL_RPC_MAX_ATTEMPTS", "2")
    monkeypatch.setenv("VANE_FTE_CONTROL_RPC_INITIAL_BACKOFF_S", "0")

    class _FlakySubmitActor:
        def __init__(self):
            self.calls = []
            self.fte_no_more_splits = _FakeRemoteMethod(self._fte_no_more_splits)

        def _fte_no_more_splits(self, task_id, source_node_id, dependency=None):
            self.calls.append(("no_more_splits", task_id, source_node_id, dependency))
            if len(self.calls) == 1:
                raise RuntimeError("temporary submit failed")
            return f"ref-{len(self.calls)}"

    monkeypatch.setattr(
        worker_handle_mod.ray, "get", lambda value, *_args, **_kwargs: {"state": "RUNNING", "ref": value}
    )
    actor = _FlakySubmitActor()
    handle = RayWorkerActorHandle(actor, memory_capacity_bytes=1 << 60)
    task_id = {"query_id": "q", "fragment_execution_id": 0, "partition_id": 1, "attempt_id": 0}

    assert handle.enqueue_fte_no_more_splits(task_id, "7") == {"state": "RUNNING", "ref": "ref-2"}
    assert actor.calls == [
        ("no_more_splits", task_id, "7", None),
        ("no_more_splits", task_id, "7", None),
    ]


def test_fte_worker_actor_handle_direct_control_does_not_retry_remote_failure(monkeypatch):
    monkeypatch.setenv("VANE_FTE_CONTROL_RPC_MAX_ATTEMPTS", "2")
    monkeypatch.setenv("VANE_FTE_CONTROL_RPC_INITIAL_BACKOFF_S", "0")

    class _FailingRemoteActor:
        def __init__(self):
            self.calls = []
            self.fte_no_more_splits = _FakeRemoteMethod(self._fte_no_more_splits)

        def _fte_no_more_splits(self, task_id, source_node_id):
            self.calls.append(("no_more_splits", task_id, source_node_id))
            return f"ref-{len(self.calls)}"

    def ray_get(_value, *_args, **_kwargs):
        raise RuntimeError("direct remote control failed")

    monkeypatch.setattr(worker_handle_mod.ray, "get", ray_get)
    actor = _FailingRemoteActor()
    handle = RayWorkerActorHandle(actor, memory_capacity_bytes=1 << 60)
    task_id = {"query_id": "q", "fragment_execution_id": 0, "partition_id": 1, "attempt_id": 0}

    with pytest.raises(RuntimeError, match="direct remote control failed"):
        handle.fte_no_more_splits(task_id, "7")

    assert actor.calls == [("no_more_splits", task_id, "7")]


def test_fte_worker_actor_handle_direct_control_retries_submission_failure(monkeypatch):
    monkeypatch.setenv("VANE_FTE_CONTROL_RPC_MAX_ATTEMPTS", "2")
    monkeypatch.setenv("VANE_FTE_CONTROL_RPC_INITIAL_BACKOFF_S", "0")

    class _FlakySubmitActor:
        def __init__(self):
            self.calls = []
            self.fte_no_more_splits = _FakeRemoteMethod(self._fte_no_more_splits)

        def _fte_no_more_splits(self, task_id, source_node_id):
            self.calls.append(("no_more_splits", task_id, source_node_id))
            if len(self.calls) == 1:
                raise RuntimeError("temporary direct submit failed")
            return f"ref-{len(self.calls)}"

    monkeypatch.setattr(
        worker_handle_mod.ray, "get", lambda value, *_args, **_kwargs: {"state": "RUNNING", "ref": value}
    )
    actor = _FlakySubmitActor()
    handle = RayWorkerActorHandle(actor, memory_capacity_bytes=1 << 60)
    task_id = {"query_id": "q", "fragment_execution_id": 0, "partition_id": 1, "attempt_id": 0}

    assert handle.fte_no_more_splits(task_id, "7") == {"state": "RUNNING", "ref": "ref-2"}
    assert actor.calls == [
        ("no_more_splits", task_id, "7"),
        ("no_more_splits", task_id, "7"),
    ]


def test_fte_drop_query_clears_fte_registry_and_worker_pressure(monkeypatch):
    monkeypatch.setattr(
        RayWorkerActorHandle,
        "_fte_task_handle_cls",
        staticmethod(lambda: _FakeFteTaskHandle),
    )
    actor0 = _FakeActor()
    actor1 = _FakeActor()
    handle0 = RayWorkerActorHandle(actor0, memory_capacity_bytes=1 << 60, worker_id="worker-0")
    handle1 = RayWorkerActorHandle(actor1, memory_capacity_bytes=1 << 60, worker_id="worker-1")

    handle0.submit_tasks(
        [
            _FakeTask(
                name="scan-drop",
                context={"query_id": "query-drop", "node_id": "7"},
                inputs={"7": {"kind": "scan_task", "data": b"drop"}},
                plan={"plan": "drop-template"},
            )
        ]
    )
    handle0.submit_tasks(
        [
            _FakeTask(
                name="scan-keep",
                context={"query_id": "query-keep", "node_id": "8"},
                inputs={"8": {"kind": "scan_task", "data": b"keep"}},
                plan={"plan": "keep-template"},
            )
        ]
    )

    before = handle0.fte_registry_stats()
    assert before["fragment_execution_count"] == 2
    assert before["partition_owner_count"] == 2
    assert before["worker_count"] == 2
    assert before["event_scheduler_count"] == 2
    assert before["event_schedulers"]["query-drop"]["event_counts"] == {
        "SplitEventsSubmitted": 1,
        "WorkerReservationCompleted": 1,
    }
    assert before["event_schedulers"]["query-drop"]["fragment_state_count"] == 1
    assert before["event_schedulers"]["query-drop"]["command_counts"] == {
        "FteCreateTaskCommand": 1,
    }
    assert sum(worker["running_attempt_count"] for worker in before["workers"].values()) == 2

    assert handle0.fte_drop_query("query-drop") == {
        "tasks_removed": 1,
        "tasks_canceled": 0,
        "fragments_removed": 2,
    }

    after = handle0.fte_registry_stats()
    assert after["fragment_execution_count"] == 1
    assert after["partition_owner_count"] == 1
    assert after["fragment_state_count"] == 1
    assert after["event_scheduler_count"] == 1
    assert sorted(after["event_schedulers"]) == ["query-keep"]
    assert after["event_schedulers"]["query-keep"]["fragment_state_count"] == 1
    assert after["event_schedulers"]["query-keep"]["command_counts"] == {
        "FteCreateTaskCommand": 1,
    }
    assert sum(worker["running_attempt_count"] for worker in after["workers"].values()) == 1
    assert all(
        "query-drop" not in attempt
        for worker in (handle0, handle1)
        for attempt in worker._fte_pressure.running_attempts
    )
    assert all(
        "query-drop" not in reservation
        for worker in (handle0, handle1)
        for reservation in worker._fte_pressure.reserved_partitions
    )
    assert ("query-drop", "query-drop:node:7") not in worker_handle_mod._FTE_FRAGMENT_EXECUTIONS
    assert ("query-keep", "query-keep:node:8") in worker_handle_mod._FTE_FRAGMENT_EXECUTIONS


def test_fte_drop_query_clears_scheduler_result_handles():
    actor = _FakeActor()
    handle = RayWorkerActorHandle(actor, memory_capacity_bytes=1 << 60)
    worker_handle_mod._FTE_RESULT_HANDLES_BY_QUERY["query-drop"] = [object()]
    worker_handle_mod._FTE_RESULT_HANDLES_BY_QUERY["query-keep"] = [object()]

    assert handle.fte_drop_query("query-drop") == {
        "tasks_removed": 1,
        "tasks_canceled": 0,
        "fragments_removed": 2,
    }

    assert "query-drop" not in worker_handle_mod._FTE_RESULT_HANDLES_BY_QUERY
    assert "query-keep" in worker_handle_mod._FTE_RESULT_HANDLES_BY_QUERY


def test_fte_drop_query_remote_failure_retains_retryable_local_query_registry(monkeypatch):
    monkeypatch.setenv("VANE_FTE_CONTROL_RPC_MAX_ATTEMPTS", "1")

    class _DeadActor(_FakeActor):
        def __init__(self):
            super().__init__()
            self.fail_once = True

        def _fte_drop_query(self, query_id):
            self.fte_calls.append(("drop_query", query_id))
            if self.fail_once:
                self.fail_once = False
                raise RuntimeError("worker actor is dead")
            return {
                "tasks_removed": 1,
                "tasks_canceled": 0,
                "fragments_removed": 2,
            }

    actor = _DeadActor()
    handle = RayWorkerActorHandle(actor, memory_capacity_bytes=1 << 60)
    handle._registered_fragment_ids = {
        "query-dead:node:1",
        "query-keep:node:2",
    }
    handle._fragment_query_ids = {
        "query-dead:node:1": "query-dead",
        "query-keep:node:2": "query-keep",
    }
    worker_handle_mod._FTE_RESULT_HANDLES_BY_QUERY["query-dead"] = [object()]
    worker_handle_mod._FTE_RESULT_HANDLES_BY_QUERY["query-keep"] = [object()]
    worker_handle_mod._FTE_SCHEDULERS.get_or_create("query-dead")

    with pytest.raises(RuntimeError, match="worker actor is dead"):
        handle.fte_drop_query("query-dead")

    assert handle._registered_fragment_ids == {
        "query-dead:node:1",
        "query-keep:node:2",
    }
    assert "query-dead" in worker_handle_mod._FTE_RESULT_HANDLES_BY_QUERY
    assert worker_handle_mod._FTE_SCHEDULERS.get("query-dead") is not None

    assert handle.fte_drop_query("query-dead") == {
        "tasks_removed": 1,
        "tasks_canceled": 0,
        "fragments_removed": 2,
    }

    assert handle._registered_fragment_ids == {"query-keep:node:2"}
    assert "query-dead" not in worker_handle_mod._FTE_RESULT_HANDLES_BY_QUERY
    assert "query-keep" in worker_handle_mod._FTE_RESULT_HANDLES_BY_QUERY
    assert worker_handle_mod._FTE_SCHEDULERS.get("query-dead") is None


def test_worker_flight_shuffle_cleanup_helper_uses_cxx_binding(monkeypatch):
    calls = []

    def _fake_require(name, hint=None):
        assert name == "cleanup_flight_shuffle_for_query"

        def _cleanup(query_id):
            calls.append((query_id, hint))
            return {
                "registry_entries_removed": 2,
                "storage_entries_removed": 7,
                "cleanup_errors": 0,
            }

        return _cleanup

    monkeypatch.setattr(worker_mod, "require_ray_cxx_attr", _fake_require)

    result = worker_mod._cleanup_flight_shuffle_for_query("query-drop")

    assert result == {
        "registry_entries_removed": 2,
        "storage_entries_removed": 7,
        "cleanup_errors": 0,
    }
    assert calls == [("query-drop", "Ensure the C++ ray extension is built with Flight shuffle cleanup support.")]


def test_fte_control_rpc_retries_transient_failure(monkeypatch):
    monkeypatch.setenv("VANE_FTE_CONTROL_RPC_MAX_ATTEMPTS", "2")
    monkeypatch.setenv("VANE_FTE_CONTROL_RPC_INITIAL_BACKOFF_S", "0")

    class _FlakyActor(_FakeActor):
        def __init__(self):
            super().__init__()
            self.fail_once = True

        def _fte_get_task_status(self, task_id):
            self.fte_calls.append(("get_status", task_id))
            if self.fail_once:
                self.fail_once = False
                raise RuntimeError("temporary ray control error")
            return {"state": "FINISHED", "task_id": task_id, "version": 4}

    actor = _FlakyActor()
    handle = RayWorkerActorHandle(actor, memory_capacity_bytes=1 << 60)
    task_id = {"query_id": "q", "fragment_execution_id": 0, "partition_id": 1, "attempt_id": 0}

    assert handle.fte_get_task_status(task_id)["state"] == "FINISHED"
    assert actor.fte_calls == [
        ("get_status", task_id),
        ("get_status", task_id),
    ]


def test_fte_control_ref_uses_async_actor_safe_get(monkeypatch):
    calls = []

    def _safe_get(ref, *, timeout=None):
        calls.append((ref, timeout))
        return "resolved"

    monkeypatch.setattr(task_control_mod, "resolve_object_refs_blocking", _safe_get)

    assert (
        task_control_mod.FteWorkerTaskControlMixin._get_fte_control_ref(
            "fte_get_task_status",
            "status-ref",
            timeout_s=7.5,
        )
        == "resolved"
    )
    assert calls == [("status-ref", 7.5)]


def test_strip_fte_dynamic_context_removes_static_bindings_only():
    context = {
        "scan_task:7": b"scan-dynamic",
        "scan_task:8": b"scan-static",
        "scan_task_nodes": "7,8",
        "exchange_source_task:3": b"exchange-dynamic",
        "exchange_source_task:4": b"exchange-static",
        "exchange_source_task_nodes": "3,4",
        "query_id": "q",
    }

    stripped = fragment_submission_mod._strip_fte_dynamic_context(
        context,
        {"7"},
        {"3"},
    )

    assert "scan_task:7" not in stripped
    assert "exchange_source_task:3" not in stripped
    assert stripped["scan_task:8"] == b"scan-static"
    assert stripped["scan_task_nodes"] == "8"
    assert stripped["exchange_source_task:4"] == b"exchange-static"
    assert stripped["exchange_source_task_nodes"] == "4"
    assert stripped["query_id"] == "q"
    assert context["scan_task_nodes"] == "7,8"


def test_fte_submit_creates_task_then_sends_split_updates(monkeypatch):
    monkeypatch.setattr(
        RayWorkerActorHandle,
        "_fte_task_handle_cls",
        staticmethod(lambda: _FakeFteTaskHandle),
    )
    actor = _FakeActor()
    handle = RayWorkerActorHandle(actor, memory_capacity_bytes=1 << 60)
    task0 = _FakeTask(
        name="scan-task-0",
        context={"query_id": "query-fte", "node_id": "7"},
        inputs={"7": {"kind": "scan_task", "data": b"a"}},
        plan={"plan": "scan-template"},
    )
    task1 = _FakeTask(
        name="scan-task-1",
        context={"query_id": "query-fte", "node_id": "7"},
        inputs={"7": {"kind": "scan_task", "data": b"b"}},
        plan={"plan": "scan-template"},
    )

    handles = handle.submit_tasks([task0, task1])

    assert isinstance(handles[0], _FakeFteTaskHandle)
    create_calls = [call for call in actor.fte_calls if call[0] == "create"]
    add_calls = [call for call in actor.fte_calls if call[0] == "add_splits"]
    assert len(handles) == 1
    assert len(create_calls) == 1
    assert add_calls == []
    assert create_calls[0][1]["task_id"]["partition_id"] == 0
    request = create_calls[0][1]
    assert request["worker_runtime"] == "fte"
    assert request["fragment_plan"] is None
    assert request["fragment_registration_result"].future().result() == {
        "registered": 1,
        "existing": 0,
        "total": 1,
    }
    assert "scan_task:7" not in request["context"]
    assert "scan_task_nodes" not in request["context"]
    assert request["dynamic_scan_source_node_ids"] == ["7"]
    assert [split["data"] for split in request["initial_splits"]["7"]] == [b"a", b"b"]
    assert [split["sequence_id"] for split in request["initial_splits"]["7"]] == [0, 1]


def test_fte_event_driven_task_source_chunks_and_drains(monkeypatch):
    monkeypatch.setenv("VANE_FTE_EVENT_SOURCE_HIGH_WATERMARK", "2")
    monkeypatch.setenv("VANE_FTE_EVENT_SOURCE_LOW_WATERMARK", "0")
    monkeypatch.setenv("VANE_FTE_EVENT_SOURCE_CHUNK_SIZE", "1")
    source_instances = []
    original_source_cls = fragment_submission_mod.FteEventDrivenTaskSource

    class _RecordingTaskSource(original_source_cls):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            source_instances.append(self)

    monkeypatch.setattr(
        fragment_submission_mod,
        "FteEventDrivenTaskSource",
        _RecordingTaskSource,
    )
    monkeypatch.setattr(
        RayWorkerActorHandle,
        "_fte_task_handle_cls",
        staticmethod(lambda: _FakeFteTaskHandle),
    )
    actor = _FakeActor()
    handle = RayWorkerActorHandle(actor, memory_capacity_bytes=1 << 60)
    tasks = [
        _FakeTask(
            name=f"scan-task-{idx}",
            context={"query_id": "query-fte-event-source", "node_id": "7"},
            inputs={"7": {"kind": "scan_task", "data": f"p{idx}".encode()}},
            plan={"plan": "scan-template"},
        )
        for idx in range(5)
    ]

    handles = handle.submit_tasks(tasks)

    assert len(handles) == 1
    create_calls = [call for call in actor.fte_calls if call[0] == "create"]
    add_calls = [call for call in actor.fte_calls if call[0] == "add_splits"]
    assert len(create_calls) == 1
    assert [split["data"] for split in create_calls[0][1]["initial_splits"]["7"]] == [b"p0", b"p1"]
    assert [split["data"] for call in add_calls for split in call[3]] == [
        b"p2",
        b"p3",
        b"p4",
    ]
    stats = handle.fte_registry_stats()["event_schedulers"]["query-fte-event-source"]
    assert stats["event_counts"]["SplitEventsSubmitted"] == 5
    assert stats["registered_task_source_count"] == 0
    assert stats["paused_task_source_count"] == 0
    assert len(source_instances) == 1
    assert source_instances[0].pause_count >= 1
    assert source_instances[0].resume_count == source_instances[0].pause_count


def test_fte_partitions_are_distributed_to_worker_owners(monkeypatch):
    monkeypatch.setattr(
        RayWorkerActorHandle,
        "_fte_task_handle_cls",
        staticmethod(lambda: _FakeFteTaskHandle),
    )
    actor0 = _FakeActor()
    actor1 = _FakeActor()
    handle0 = RayWorkerActorHandle(actor0, memory_capacity_bytes=1 << 60, worker_id="worker-0")
    handle1 = RayWorkerActorHandle(actor1, memory_capacity_bytes=1 << 60, worker_id="worker-1")
    task0 = _FakeTask(
        name="scan-task-0",
        context={"query_id": "query-fte-owner", "node_id": "7"},
        inputs={"7": {"kind": "scan_task", "data": b"a"}},
        plan={"plan": "scan-template"},
    )
    task1 = _FakeTask(
        name="scan-task-1",
        context={"query_id": "query-fte-owner", "node_id": "7"},
        inputs={"7": {"kind": "scan_task", "data": b"b"}},
        plan={"plan": "scan-template"},
    )

    first = handle0.submit_tasks([task0])
    second = handle1.submit_tasks([task1])

    assert isinstance(first[0], _FakeFteTaskHandle)
    assert second == []
    assert [call[0] for call in actor0.fte_calls] == [
        "create",
        "wait_split_queue",
        "add_splits",
    ]
    assert actor1.fte_calls == []
    assert actor0.fte_calls[0][1]["initial_splits"]["7"][0]["data"] == b"a"
    assert actor0.fte_calls[1][2] == "7"
    assert actor0.fte_calls[2][2] == "7"
    assert actor0.fte_calls[2][3][0]["data"] == b"b"
    assert first[0].worker_handle is handle0


def test_fte_owner_selection_uses_worker_split_pressure(monkeypatch):
    monkeypatch.setattr(
        RayWorkerActorHandle,
        "_fte_task_handle_cls",
        staticmethod(lambda: _FakeFteTaskHandle),
    )
    monkeypatch.setattr(
        fragment_submission_mod,
        "_split_exchange_source_task_by_partition",
        lambda value: ([(int(value[-1:]), value)], 2, 2, False),
    )
    actor0 = _FakeActor()
    actor1 = _FakeActor()
    handle0 = RayWorkerActorHandle(actor0, memory_capacity_bytes=1 << 60, worker_id="worker-0")
    handle1 = RayWorkerActorHandle(actor1, memory_capacity_bytes=1 << 60, worker_id="worker-1")
    task0 = _FakeTask(
        name="exchange-task-0",
        context={"query_id": "query-fte-pressure", "node_id": "8"},
        inputs={"3": {"kind": "exchange_source_task", "data": b"p0"}},
        plan={"plan": "exchange-template"},
    )
    task1 = _FakeTask(
        name="exchange-task-1",
        context={"query_id": "query-fte-pressure", "node_id": "8"},
        inputs={"3": {"kind": "exchange_source_task", "data": b"p1"}},
        plan={"plan": "exchange-template"},
    )

    handles = handle0.submit_tasks([task0, task1])

    assert [handle.worker_handle for handle in handles] == [handle0, handle1]
    assert [call[1]["task_id"]["partition_id"] for call in actor0.fte_calls if call[0] == "create"] == [0]
    assert [call[1]["task_id"]["partition_id"] for call in actor1.fte_calls if call[0] == "create"] == [1]
    assert handle0.fte_pressure_stats()["running_attempt_count"] == 1
    assert handle1.fte_pressure_stats()["running_attempt_count"] == 1
    assert handle0.fte_pressure_stats()["assigned_split_bytes"] == len(b"p0")
    assert handle1.fte_pressure_stats()["assigned_split_bytes"] == len(b"p1")
    assert handle0.fte_registry_stats()["partition_owner_count"] == 2


def test_fte_registry_stats_reports_query_stage_partition_metrics(monkeypatch):
    monkeypatch.setattr(
        RayWorkerActorHandle,
        "_fte_task_handle_cls",
        staticmethod(lambda: _FakeFteTaskHandle),
    )
    actor = _FakeActor()
    handle = RayWorkerActorHandle(actor, memory_capacity_bytes=1 << 60, worker_id="worker-0")
    task = _FakeTask(
        name="scan-task-metrics",
        context={"query_id": "query-fte-metrics", "node_id": "7"},
        inputs={"7": {"kind": "scan_task", "data": b"a"}},
        plan={"plan": "scan-template"},
    )

    task_handles = handle.submit_tasks([task])

    stats = handle.fte_registry_stats()
    query = stats["queries"]["query-fte-metrics"]
    stage = query["fragment_executions"]["query-fte-metrics:node:7"]
    partition = stage["partitions"]["0"]

    assert [str(task_handle.task_id) for task_handle in task_handles] == ["query-fte-metrics.0.0.0"]
    assert query["fragment_execution_count"] == 1
    assert query["partition_count"] == 1
    assert query["running_count"] == 1
    assert query["waiting_for_node_count"] == 0
    assert stage["running_count"] == 1
    assert stage["execution_class_counts"] == {"STANDARD": 1}
    assert partition["state"] == "RUNNING"
    assert partition["owner_worker_id"] == "worker-0"
    assert partition["initial_split_count_by_source"] == {"7": 1}
    assert partition["no_more_splits"] == []
    assert partition["running_attempts"][0]["attempt_id"] == "query-fte-metrics.0.0.0"
    assert partition["running_attempts"][0]["worker_id"] == "worker-0"

    handle.handle_fte_task_status(
        {
            "state": "FINISHED",
            "task_id": task_handles[0].task_id.to_dict(),
            "version": 1,
        }
    )

    finished_stats = handle.fte_registry_stats()
    finished_partition = finished_stats["queries"]["query-fte-metrics"]["fragment_executions"][
        "query-fte-metrics:node:7"
    ]["partitions"]["0"]
    assert finished_stats["queries"]["query-fte-metrics"]["finished"] is True
    assert finished_partition["state"] == "FINISHED"
    assert finished_partition["running_attempts"] == []
    assert finished_partition["selected_attempt"] == 0
    assert finished_partition["finished_attempts"] == [0]


def test_fte_owner_selection_uses_reserved_memory_pressure(monkeypatch):
    actor0 = _FakeActor()
    actor1 = _FakeActor()
    high_memory = RayWorkerActorHandle(actor0, memory_capacity_bytes=20, worker_id="worker-0")
    low_memory = RayWorkerActorHandle(actor1, memory_capacity_bytes=20, worker_id="worker-1")

    high_memory.reserve_fte_partition(
        "query-pressure",
        "fragment",
        0,
        memory_requirement_bytes=15,
    )
    low_memory.reserve_fte_partition(
        "query-pressure",
        "fragment",
        1,
        memory_requirement_bytes=5,
    )

    selected = high_memory._select_fte_worker(memory_requirement_bytes=10)

    assert selected is low_memory
    assert high_memory.fte_pressure_stats()["reserved_memory_bytes"] == 15
    assert high_memory.fte_pressure_stats()["total_memory_bytes"] == 15
    high_memory.release_fte_partition_reservation("query-pressure", "fragment", 0)
    assert high_memory.fte_pressure_stats()["reserved_memory_bytes"] == 0
    assert high_memory.fte_pressure_stats()["total_memory_bytes"] == 0


def test_ray_worker_handle_requires_positive_ray_memory_capacity():
    actor = _FakeActor()

    with pytest.raises(TypeError):
        RayWorkerActorHandle(actor)
    with pytest.raises(ValueError, match="memory_capacity_bytes must be positive"):
        RayWorkerActorHandle(actor, memory_capacity_bytes=0)


def test_fte_empty_worker_rejects_task_larger_than_ray_memory_capacity():
    handle = RayWorkerActorHandle(
        _FakeActor(),
        worker_id="worker-0",
        memory_capacity_bytes=9,
    )

    assert not worker_handle_mod._fte_worker_has_memory_capacity(
        handle,
        memory_requirement_bytes=10,
    )


def test_fte_existing_owner_rechecks_memory_capacity_and_reselects(monkeypatch):
    actor0 = _FakeActor()
    actor1 = _FakeActor()
    owner = RayWorkerActorHandle(actor0, memory_capacity_bytes=25, worker_id="worker-0")
    replacement = RayWorkerActorHandle(actor1, memory_capacity_bytes=25, worker_id="worker-1")
    query_id = "query-owner-capacity"
    fragment_id = _install_manual_test_fragment(query_id, "7")
    owner_key = (query_id, fragment_id, 0)
    worker_handle_mod._FTE_PARTITION_OWNERS[owner_key] = owner
    owner.reserve_fte_partition(
        "query-other",
        "fragment",
        1,
        memory_requirement_bytes=20,
    )

    reservation = owner._fte_worker_placement_manager.acquire(
        query_id=query_id,
        fragment_id=fragment_id,
        partition_id=0,
        memory_requirement_bytes=10,
    )

    assert reservation.worker is replacement
    assert worker_handle_mod._FTE_PARTITION_OWNERS[owner_key] is replacement
    assert owner.fte_pressure_stats()["reserved_memory_bytes"] == 20
    assert replacement.fte_pressure_stats()["reserved_memory_bytes"] == 10


def test_fte_existing_owner_capacity_failure_clears_owner(monkeypatch):
    actor = _FakeActor()
    owner = RayWorkerActorHandle(actor, memory_capacity_bytes=25, worker_id="worker-0")
    query_id = "query-owner-no-replacement"
    fragment_id = _install_manual_test_fragment(query_id, "7")
    owner_key = (query_id, fragment_id, 0)
    worker_handle_mod._FTE_PARTITION_OWNERS[owner_key] = owner
    owner.reserve_fte_partition(
        query_id,
        fragment_id,
        0,
        memory_requirement_bytes=10,
    )
    owner.reserve_fte_partition(
        "query-other",
        "fragment",
        1,
        memory_requirement_bytes=20,
    )

    with pytest.raises(Exception, match="reservation available"):
        owner._fte_worker_placement_manager.acquire(
            query_id=query_id,
            fragment_id=fragment_id,
            partition_id=0,
            memory_requirement_bytes=10,
        )

    assert owner_key not in worker_handle_mod._FTE_PARTITION_OWNERS
    assert owner.fte_pressure_stats()["reserved_memory_bytes"] == 20


def test_fte_reservation_failure_does_not_publish_owner(monkeypatch):
    actor = _FakeActor()
    handle = RayWorkerActorHandle(actor, memory_capacity_bytes=1 << 60, worker_id="worker-0")
    query_id = "query-reservation-failure"
    fragment_id = _install_manual_test_fragment(query_id, "7")
    owner_key = (query_id, fragment_id, 0)

    def _raise_reservation_error(*_args, **_kwargs):
        raise RuntimeError("reserve exploded")

    monkeypatch.setattr(handle, "reserve_fte_partition", _raise_reservation_error)

    with pytest.raises(RuntimeError, match="reserve exploded"):
        handle._fte_worker_placement_manager.acquire(
            query_id=query_id,
            fragment_id=fragment_id,
            partition_id=0,
            memory_requirement_bytes=10,
        )
    assert owner_key not in worker_handle_mod._FTE_PARTITION_OWNERS

    worker_handle_mod._FTE_PARTITION_OWNERS[owner_key] = handle
    with pytest.raises(RuntimeError, match="reserve exploded"):
        handle._fte_worker_placement_manager.acquire(
            query_id=query_id,
            fragment_id=fragment_id,
            partition_id=0,
            memory_requirement_bytes=10,
        )
    assert owner_key not in worker_handle_mod._FTE_PARTITION_OWNERS


def test_fte_owner_selection_prefers_node_requirement_host(monkeypatch):
    actor0 = _FakeActor()
    actor1 = _FakeActor()
    non_matching = RayWorkerActorHandle(actor0, memory_capacity_bytes=1 << 60, worker_id="aaa#0")
    matching = RayWorkerActorHandle(actor1, memory_capacity_bytes=1 << 60, worker_id="zzz#0")

    selected = non_matching._select_fte_worker(
        node_requirements=NodeRequirements(host="zzz"),
    )

    assert selected is matching


def test_fte_non_remote_node_requirement_requires_matching_host(monkeypatch):
    actor0 = _FakeActor()
    actor1 = _FakeActor()
    non_matching = RayWorkerActorHandle(actor0, memory_capacity_bytes=1 << 60, worker_id="aaa#0")
    matching = RayWorkerActorHandle(actor1, memory_capacity_bytes=1 << 60, worker_id="bbb#0")

    selected = non_matching._select_fte_worker(
        node_requirements=NodeRequirements(host="bbb", remotely_accessible=False),
    )
    missing = non_matching._select_fte_worker(
        node_requirements=NodeRequirements(host="missing", remotely_accessible=False),
    )

    assert selected is matching
    assert missing is None


def test_fte_remote_node_requirement_waits_before_fallback(monkeypatch):
    monkeypatch.setenv("VANE_FTE_EXHAUSTED_NODE_WAIT_PERIOD_S", "60")
    actor0 = _FakeActor()
    actor1 = _FakeActor()
    fallback = RayWorkerActorHandle(actor0, memory_capacity_bytes=15, worker_id="aaa#0")
    preferred = RayWorkerActorHandle(actor1, memory_capacity_bytes=15, worker_id="bbb#0")
    preferred.reserve_fte_partition(
        "query-locality",
        "fragment",
        0,
        memory_requirement_bytes=10,
    )

    not_expired = fallback._select_fte_worker(
        memory_requirement_bytes=10,
        node_requirements=NodeRequirements(host="bbb"),
        node_requirements_wait_started_at=worker_handle_mod.time.time(),
    )
    expired = fallback._select_fte_worker(
        memory_requirement_bytes=10,
        node_requirements=NodeRequirements(host="bbb"),
        node_requirements_wait_started_at=worker_handle_mod.time.time() - 61,
    )

    assert not_expired is None
    assert expired is fallback


def test_fte_non_remote_node_requirement_never_fallback_after_wait(monkeypatch):
    monkeypatch.setenv("VANE_FTE_EXHAUSTED_NODE_WAIT_PERIOD_S", "0")
    actor0 = _FakeActor()
    actor1 = _FakeActor()
    fallback = RayWorkerActorHandle(actor0, memory_capacity_bytes=15, worker_id="aaa#0")
    preferred = RayWorkerActorHandle(actor1, memory_capacity_bytes=15, worker_id="bbb#0")
    preferred.reserve_fte_partition(
        "query-locality-hard",
        "fragment",
        0,
        memory_requirement_bytes=10,
    )

    selected = fallback._select_fte_worker(
        memory_requirement_bytes=10,
        node_requirements=NodeRequirements(host="bbb", remotely_accessible=False),
        node_requirements_wait_started_at=worker_handle_mod.time.time() - 60,
    )

    assert selected is None


def test_fte_no_matching_node_waits_before_fail(monkeypatch):
    monkeypatch.setenv("VANE_FTE_ALLOWED_NO_MATCHING_NODE_PERIOD_S", "60")
    actor = _FakeActor()
    handle = RayWorkerActorHandle(actor, memory_capacity_bytes=1 << 60, worker_id="aaa#0")
    stage = handle._get_or_create_fte_fragment_execution(
        {
            "query_id": "query-no-matching-wait",
            "fragment_id": "query-no-matching-wait:node:7",
            "cfg": {"cfg": "scan"},
            "context": {},
            "task_context_info": {},
        },
        dynamic_scan_sources={"7"},
        dynamic_exchange_sources=set(),
    )

    scheduled = stage.apply_assignment_result(
        AssignmentResult(
            partitions_added=[PartitionInfo(0, NodeRequirements(host="missing", remotely_accessible=False))],
            sealed_partitions=[0],
        )
    )

    assert scheduled == []
    assert actor.fte_calls == []
    assert stage.partitions[0].no_matching_node_started_at is not None
    assert stage.partitions[0].failed is False


def test_fte_no_matching_node_period_expiry_fails_query(monkeypatch):
    monkeypatch.setenv("VANE_FTE_ALLOWED_NO_MATCHING_NODE_PERIOD_S", "0")
    actor = _FakeActor()
    handle = RayWorkerActorHandle(actor, memory_capacity_bytes=1 << 60, worker_id="aaa#0")
    stage = handle._get_or_create_fte_fragment_execution(
        {
            "query_id": "query-no-matching-expired",
            "fragment_id": "query-no-matching-expired:node:7",
            "cfg": {"cfg": "scan"},
            "context": {},
            "task_context_info": {},
        },
        dynamic_scan_sources={"7"},
        dynamic_exchange_sources=set(),
    )

    with pytest.raises(RuntimeError, match="No nodes available to run query"):
        stage.apply_assignment_result(
            AssignmentResult(
                partitions_added=[PartitionInfo(0, NodeRequirements(host="missing", remotely_accessible=False))],
                sealed_partitions=[0],
            )
        )


def test_fte_strict_worker_reservation_returns_pending_handle_until_capacity_frees(monkeypatch):
    monkeypatch.setattr(
        RayWorkerActorHandle,
        "_fte_task_handle_cls",
        staticmethod(lambda: _FakeFteTaskHandle),
    )
    monkeypatch.setattr(
        fragment_submission_mod,
        "_split_exchange_source_task_by_partition",
        lambda value: ([(int(value[-1:]), value)], 2, 2, False),
    )
    actor = _FakeActor()
    handle = RayWorkerActorHandle(actor, memory_capacity_bytes=15, worker_id="worker-0")
    task0 = _FakeTask(
        name="exchange-task-0",
        context={"query_id": "query-strict-reservation", "node_id": "8"},
        inputs={"3": {"kind": "exchange_source_task", "data": b"p0"}},
        plan={"plan": "exchange-template"},
    )
    task1 = _FakeTask(
        name="exchange-task-1",
        context={"query_id": "query-strict-reservation", "node_id": "8"},
        inputs={"3": {"kind": "exchange_source_task", "data": b"p1"}},
        plan={"plan": "exchange-template"},
    )

    handles = handle.submit_tasks([task0, task1])

    assert len(handles) == 1
    assert isinstance(handles[0], _FakeFteTaskHandle)
    assert [call[1]["task_id"]["partition_id"] for call in actor.fte_calls if call[0] == "create"] == [0]
    pending_key = ("query-strict-reservation", "query-strict-reservation:node:8", 1)
    pending_future = worker_handle_mod._FTE_PENDING_WORKER_RESERVATIONS[pending_key]
    assert pending_future.done() is False
    assert pending_future.cancelled() is False
    assert handle.fte_registry_stats()["pending_worker_reservation_count"] == 1
    assert (
        handle._handles_for_worker_reservation_completed_event(
            WorkerReservationCompleted(
                "query-strict-reservation",
                0,
                "query-strict-reservation:node:8",
                1,
                0,
                "worker-0",
            )
        )
        == []
    )
    assert worker_handle_mod._FTE_PENDING_WORKER_RESERVATIONS[pending_key] is pending_future
    assert [str(task_handle.task_id) for task_handle in handle.pop_fte_result_handles("query-strict-reservation")] == [
        "query-strict-reservation.0.0.0"
    ]
    assert handle.pop_fte_result_handles("query-strict-reservation") == []

    handle.record_fte_task_terminal(handles[0].task_id)
    scheduled = handle.pop_fte_result_handles("query-strict-reservation")

    assert len(scheduled) == 1
    assert isinstance(scheduled[0], _FakeFteTaskHandle)
    assert scheduled[0].task_id.partition_id == 1
    assert pending_future.done() is True
    assert pending_key not in worker_handle_mod._FTE_PENDING_WORKER_RESERVATIONS
    assert handle.fte_registry_stats()["pending_worker_reservation_count"] == 0
    assert [call[1]["task_id"]["partition_id"] for call in actor.fte_calls if call[0] == "create"] == [0, 1]


def _submit_strict_worker_reservation_pending_pair(monkeypatch, query_id: str):
    monkeypatch.setattr(
        RayWorkerActorHandle,
        "_fte_task_handle_cls",
        staticmethod(lambda: _FakeFteTaskHandle),
    )
    monkeypatch.setattr(
        fragment_submission_mod,
        "_split_exchange_source_task_by_partition",
        lambda value: ([(int(value[-1:]), value)], 2, 2, False),
    )
    actor = _FakeActor()
    handle = RayWorkerActorHandle(actor, memory_capacity_bytes=15, worker_id="worker-0")
    tasks = [
        _FakeTask(
            name="exchange-task-0",
            context={"query_id": query_id, "node_id": "8"},
            inputs={"3": {"kind": "exchange_source_task", "data": b"p0"}},
            plan={"plan": "exchange-template"},
        ),
        _FakeTask(
            name="exchange-task-1",
            context={"query_id": query_id, "node_id": "8"},
            inputs={"3": {"kind": "exchange_source_task", "data": b"p1"}},
            plan={"plan": "exchange-template"},
        ),
    ]
    handles = handle.submit_tasks(tasks)
    fragment_id = f"{query_id}:node:8"
    pending_key = (query_id, fragment_id, 1)
    pending_future = worker_handle_mod._FTE_PENDING_WORKER_RESERVATIONS[pending_key]
    assert len(handles) == 1
    assert [call[1]["task_id"]["partition_id"] for call in actor.fte_calls if call[0] == "create"] == [0]
    return actor, handle, pending_key, pending_future


def _completed_test_reservation(future, worker):
    return worker_handle_mod.FteWorkerReservation(
        query_id=future.query_id,
        fragment_execution_id=future.fragment_execution_id,
        fragment_id=future.fragment_id,
        partition_id=future.partition_id,
        worker=worker,
        stage_id=fte_stage_id_for_fragment(future.query_id, future.fragment_id),
        task_lease_id=f"test-lease-{future.reservation_generation}",
        attempt_id=f"{future.query_id}.{future.fragment_execution_id}.{future.partition_id}.0",
    )


def test_fte_completed_worker_reservation_reselects_when_event_worker_failed(monkeypatch):
    query_id = "query-reservation-worker-failed"
    actor0, failed, pending_key, pending_future = _submit_strict_worker_reservation_pending_pair(
        monkeypatch,
        query_id,
    )
    actor1 = _FakeActor()
    replacement = RayWorkerActorHandle(actor1, memory_capacity_bytes=15, worker_id="worker-1")
    failed._fte_healthy = False

    pending_future.set_result(_completed_test_reservation(pending_future, failed))
    scheduler = worker_handle_mod._FTE_SCHEDULERS.get(query_id)
    assert scheduler is not None

    scheduled = scheduler.drain()

    assert len(scheduled) == 1
    assert scheduled[0].worker_handle is replacement
    assert pending_key not in worker_handle_mod._FTE_PENDING_WORKER_RESERVATIONS
    assert worker_handle_mod._FTE_PARTITION_OWNERS[pending_key] is replacement
    assert [call[1]["task_id"]["partition_id"] for call in actor0.fte_calls if call[0] == "create"] == [0]
    assert [call[1]["task_id"]["partition_id"] for call in actor1.fte_calls if call[0] == "create"] == [1]


def test_fte_worker_reservation_completion_after_query_drop_is_ignored(monkeypatch):
    query_id = "query-reservation-after-drop"
    actor, handle, pending_key, pending_future = _submit_strict_worker_reservation_pending_pair(
        monkeypatch,
        query_id,
    )
    pending_future.set_result(_completed_test_reservation(pending_future, handle))
    scheduler = worker_handle_mod._FTE_SCHEDULERS.get(query_id)
    assert scheduler is not None
    assert scheduler.stats().to_dict()["queued_events"] == 1

    handle.fte_drop_query(query_id)

    assert scheduler.drain() == []
    assert pending_key not in worker_handle_mod._FTE_PENDING_WORKER_RESERVATIONS
    assert (query_id, pending_key[1]) not in worker_handle_mod._FTE_FRAGMENT_EXECUTIONS
    assert [call[1]["task_id"]["partition_id"] for call in actor.fte_calls if call[0] == "create"] == [0]


def test_fte_stale_worker_reservation_generation_does_not_consume_new_future(monkeypatch):
    query_id = "query-reservation-stale-generation"
    actor0, handle, pending_key, old_future = _submit_strict_worker_reservation_pending_pair(
        monkeypatch,
        query_id,
    )
    actor1 = _FakeActor()
    replacement = RayWorkerActorHandle(actor1, memory_capacity_bytes=15, worker_id="worker-1")
    old_future.set_result(_completed_test_reservation(old_future, handle))
    with worker_handle_mod._FTE_REGISTRY_LOCK:
        assert worker_handle_mod._FTE_PENDING_WORKER_RESERVATIONS.pop(pending_key) is old_future
        stage = worker_handle_mod._FTE_FRAGMENT_EXECUTIONS[(query_id, pending_key[1])]
    partition = stage.partitions[1]
    new_future, created = handle._fte_worker_placement_manager.request_async(
        query_id=query_id,
        fragment_execution_id=stage.fragment_execution_id,
        fragment_id=pending_key[1],
        partition_id=1,
        memory_requirement_bytes=partition.memory_requirement_bytes,
        execution_class=partition.execution_class,
        node_requirements=partition.node_requirements,
        node_requirements_wait_started_at=partition.node_wait_started_at,
        on_done=handle._enqueue_fte_worker_reservation_completion,
    )
    scheduler = worker_handle_mod._FTE_SCHEDULERS.get(query_id)
    assert scheduler is not None
    assert created is True
    assert new_future.reservation_generation == old_future.reservation_generation + 1

    assert scheduler.drain() == []
    assert worker_handle_mod._FTE_PENDING_WORKER_RESERVATIONS[pending_key] is new_future
    assert [call[1]["task_id"]["partition_id"] for call in actor0.fte_calls if call[0] == "create"] == [0]

    new_future.set_result(_completed_test_reservation(new_future, replacement))
    scheduled = scheduler.drain()

    assert len(scheduled) == 1
    assert scheduled[0].worker_handle is replacement
    assert pending_key not in worker_handle_mod._FTE_PENDING_WORKER_RESERVATIONS
    assert [call[1]["task_id"]["partition_id"] for call in actor1.fte_calls if call[0] == "create"] == [1]


def test_fte_pending_worker_reservation_cancelled_on_query_drop(monkeypatch):
    monkeypatch.setattr(
        RayWorkerActorHandle,
        "_fte_task_handle_cls",
        staticmethod(lambda: _FakeFteTaskHandle),
    )
    monkeypatch.setattr(
        fragment_submission_mod,
        "_split_exchange_source_task_by_partition",
        lambda value: ([(int(value[-1:]), value)], 2, 2, False),
    )
    actor = _FakeActor()
    handle = RayWorkerActorHandle(actor, memory_capacity_bytes=15, worker_id="worker-0")
    tasks = [
        _FakeTask(
            name="exchange-task-0",
            context={"query_id": "query-drop-pending-reservation", "node_id": "8"},
            inputs={"3": {"kind": "exchange_source_task", "data": b"p0"}},
            plan={"plan": "exchange-template"},
        ),
        _FakeTask(
            name="exchange-task-1",
            context={"query_id": "query-drop-pending-reservation", "node_id": "8"},
            inputs={"3": {"kind": "exchange_source_task", "data": b"p1"}},
            plan={"plan": "exchange-template"},
        ),
    ]

    handles = handle.submit_tasks(tasks)
    pending_key = ("query-drop-pending-reservation", "query-drop-pending-reservation:node:8", 1)
    pending_future = worker_handle_mod._FTE_PENDING_WORKER_RESERVATIONS[pending_key]

    result = handle.fte_drop_query("query-drop-pending-reservation")

    assert len(handles) == 1
    assert result == {"tasks_removed": 1, "tasks_canceled": 0, "fragments_removed": 2}
    assert pending_future.cancelled() is True
    assert pending_key not in worker_handle_mod._FTE_PENDING_WORKER_RESERVATIONS
    assert (
        "query-drop-pending-reservation",
        "query-drop-pending-reservation:node:8",
    ) not in worker_handle_mod._FTE_FRAGMENT_EXECUTIONS


def test_fte_denied_descriptor_is_not_registered_and_block_is_removed_when_abandoned():
    query_id = "query-resource-waiter"
    fragment_id = _install_manual_test_fragment(query_id, "8")
    manager = get_query_resource_manager(query_id)
    manager.update_allocation(
        QueryAllocation(
            resources=ResourceVector(),
            node_allocations=(),
            actor_placements=(),
            generation=2,
        ),
        admission_open=False,
    )

    with pytest.raises(FteWorkerReservationUnavailable) as exc_info:
        worker_handle_mod._acquire_fte_partition_task_lease(
            query_id=query_id,
            fragment_execution_id=7,
            fragment_id=fragment_id,
            partition_id=0,
            node_id="node-a",
        )

    assert exc_info.value.blocked_reason == "allocation_pending"
    stage_id = fte_stage_id_for_fragment(query_id, fragment_id)
    assert manager.snapshot()["stages"][stage_id]["pending_task_count"] == 0
    stats = worker_handle_mod.fte_registry_stats()
    assert stats["partition_task_waiter_count"] == 0
    assert stats["stage_submission_block_count"] == 1

    worker_handle_mod.FteWorkerPlacementManager.release_owner(
        query_id=query_id,
        fragment_id=fragment_id,
        partition_id=0,
    )

    assert manager.snapshot()["stages"][stage_id]["pending_task_count"] == 0
    stats = worker_handle_mod.fte_registry_stats()
    assert stats["partition_task_waiter_count"] == 0
    assert stats["stage_submission_block_count"] == 0


def test_fte_worker_capacity_tracks_all_node_waiters_without_a_second_cap(monkeypatch):
    monkeypatch.setattr(
        RayWorkerActorHandle,
        "_fte_task_handle_cls",
        staticmethod(lambda: _FakeFteTaskHandle),
    )
    monkeypatch.setattr(
        fragment_submission_mod,
        "_split_exchange_source_task_by_partition",
        lambda value: ([(int(value[-1:]), value)], 3, 3, False),
    )
    actor = _FakeActor()
    handle = RayWorkerActorHandle(actor, memory_capacity_bytes=15, worker_id="worker-0")

    running = handle.submit_tasks(
        [
            _FakeTask(
                name="running-standard",
                context={"query_id": "query-wait-cap", "node_id": "8"},
                inputs={"3": {"kind": "exchange_source_task", "data": b"p0"}},
                plan={"plan": "exchange-template"},
            )
        ]
    )[0]
    first_pending = handle.submit_tasks(
        [
            _FakeTask(
                name="first-pending-standard",
                context={"query_id": "query-wait-cap", "node_id": "8"},
                inputs={"3": {"kind": "exchange_source_task", "data": b"p1"}},
                plan={"plan": "exchange-template"},
            )
        ]
    )
    second_pending = handle.submit_tasks(
        [
            _FakeTask(
                name="second-pending-standard",
                context={"query_id": "query-wait-cap", "node_id": "8"},
                inputs={"3": {"kind": "exchange_source_task", "data": b"p2"}},
                plan={"plan": "exchange-template"},
            )
        ]
    )

    assert isinstance(running, _FakeFteTaskHandle)
    assert first_pending == []
    assert second_pending == []
    assert [str(task_handle.task_id) for task_handle in handle.pop_fte_result_handles("query-wait-cap")] == [
        "query-wait-cap.0.0.0"
    ]
    stage = next(
        stage
        for (query_id, _), stage in worker_handle_mod._FTE_FRAGMENT_EXECUTIONS.items()
        if query_id == "query-wait-cap"
    )
    assert stage.waiting_for_node_count() == 2
    assert stage.partitions[1].node_wait_started_at is not None
    assert stage.partitions[2].node_wait_started_at is not None
    assert handle.pop_fte_result_handles("query-wait-cap") == []
    assert stage.partitions[2].node_wait_started_at is not None

    handle.record_fte_task_terminal(running.task_id)
    scheduled = handle.pop_fte_result_handles("query-wait-cap")

    assert [str(task_handle.task_id) for task_handle in scheduled] == ["query-wait-cap.0.1.0"]
    assert stage.partitions[1].node_wait_started_at is None
    assert stage.partitions[2].node_wait_started_at is not None
    assert stage.waiting_for_node_count() == 1
    assert [call[1]["task_id"]["partition_id"] for call in actor.fte_calls if call[0] == "create"] == [0, 1]


def test_fte_dynamic_exchange_running_window_defers_extra_partitions(monkeypatch):
    _register_test_query_graph(
        "query-dynamic-window",
        ["query-dynamic-window:node:8"],
        max_concurrency=1,
    )
    monkeypatch.setattr(
        RayWorkerActorHandle,
        "_fte_task_handle_cls",
        staticmethod(lambda: _FakeFteTaskHandle),
    )
    monkeypatch.setattr(
        fragment_submission_mod,
        "_split_exchange_source_task_by_partition",
        lambda value: ([(int(value[-1:]), value)], 2, 2, False),
    )
    actor = _FakeActor()
    handle = RayWorkerActorHandle(actor, memory_capacity_bytes=100, worker_id="worker-0")
    tasks = [
        _FakeTask(
            name="exchange-task-0",
            context={"query_id": "query-dynamic-window", "node_id": "8"},
            inputs={"3": {"kind": "exchange_source_task", "data": b"p0"}},
            plan={"plan": "exchange-template"},
        ),
        _FakeTask(
            name="exchange-task-1",
            context={"query_id": "query-dynamic-window", "node_id": "8"},
            inputs={"3": {"kind": "exchange_source_task", "data": b"p1"}},
            plan={"plan": "exchange-template"},
        ),
    ]

    handles = handle.submit_tasks(tasks)

    assert [str(task_handle.task_id) for task_handle in handles] == ["query-dynamic-window.0.0.0"]
    stage = next(
        stage
        for (query_id, _), stage in worker_handle_mod._FTE_FRAGMENT_EXECUTIONS.items()
        if query_id == "query-dynamic-window"
    )
    assert stage.waiting_for_node_count() == 0
    assert stage.partitions[1].ready_for_scheduling is False
    assert stage.partitions[1].execution_ready_deferred is True
    assert stage.partitions[1].node_wait_started_at is None
    stage_id = fte_stage_id_for_fragment(
        "query-dynamic-window",
        stage.fragment_id,
    )
    assert get_query_resource_manager("query-dynamic-window").snapshot()["stages"][stage_id]["pending_task_count"] == 0
    assert [call[1]["task_id"]["partition_id"] for call in actor.fte_calls if call[0] == "create"] == [0]
    assert [str(task_handle.task_id) for task_handle in handle.pop_fte_result_handles("query-dynamic-window")] == [
        "query-dynamic-window.0.0.0"
    ]

    scheduled = handle.handle_fte_task_status(
        {
            "state": "FINISHED",
            "task_id": handles[0].task_id.to_dict(),
            "version": 1,
        }
    )

    assert [str(task_handle.task_id) for task_handle in scheduled] == ["query-dynamic-window.0.1.0"]
    assert [call[1]["task_id"]["partition_id"] for call in actor.fte_calls if call[0] == "create"] == [0, 1]


def test_fte_submission_window_keeps_36_descriptors_but_only_7_in_qrm(monkeypatch):
    query_id = "query-36-descriptors-7-running"
    fragment_id = f"{query_id}:node:8"
    _register_test_query_graph(query_id, [fragment_id], max_concurrency=7)
    monkeypatch.setattr(
        fragment_submission_mod,
        "_split_exchange_source_task_by_partition",
        lambda value: ([(int(value[1:]), value)], 36, 36, False),
    )
    actor = _FakeActor()
    handle = RayWorkerActorHandle(
        actor,
        memory_capacity_bytes=1_000,
        worker_id="worker-window",
    )
    tasks = [
        _FakeTask(
            name=f"producer-{partition_id}",
            context={"query_id": query_id, "node_id": "8"},
            inputs={
                "3": {
                    "kind": "exchange_source_task",
                    "data": f"p{partition_id}".encode(),
                }
            },
            plan={"plan": "exchange-template"},
        )
        for partition_id in range(36)
    ]

    handles = handle.submit_tasks(tasks)
    stage = worker_handle_mod._FTE_FRAGMENT_EXECUTIONS[(query_id, fragment_id)]
    stage_id = fte_stage_id_for_fragment(query_id, fragment_id)
    manager = get_query_resource_manager(query_id)
    snapshot = manager.snapshot()
    stats = handle.fte_registry_stats()

    assert len(stage.partitions) == 36
    assert [attempt.task_id.partition_id for attempt in handles] == list(range(7))
    assert snapshot["stages"][stage_id]["active_task_count"] == 7
    assert snapshot["stages"][stage_id]["pending_task_count"] == 0
    assert stats["partition_task_waiter_count"] == 0
    assert stats["pending_worker_reservation_count"] == 0
    assert stats["stage_submission_probe_count"] == 0
    assert stats["stage_submission_block_count"] == 1
    assert sum(partition.execution_ready_deferred for partition in stage.partitions.values()) == 29
    assert (
        sum(
            partition.node_wait_started_at is not None
            for partition in stage.partitions.values()
            if not partition.running_attempts
        )
        == 0
    )

    refill = handle.handle_fte_task_status(
        {
            "state": "FINISHED",
            "task_id": handles[0].task_id.to_dict(),
            "version": 1,
        }
    )
    snapshot = manager.snapshot()
    stats = handle.fte_registry_stats()

    assert [attempt.task_id.partition_id for attempt in refill] == [7]
    assert snapshot["stages"][stage_id]["active_task_count"] == 7
    assert snapshot["stages"][stage_id]["pending_task_count"] == 0
    assert stats["partition_task_waiter_count"] == 0
    assert stats["pending_worker_reservation_count"] == 0
    assert sum(partition.execution_ready_deferred for partition in stage.partitions.values()) == 28


def test_fte_worker_capacity_registers_every_ready_partition_with_credit_authority(monkeypatch):
    monkeypatch.setattr(
        RayWorkerActorHandle,
        "_fte_task_handle_cls",
        staticmethod(lambda: _FakeFteTaskHandle),
    )
    actor = _FakeActor()
    handle = RayWorkerActorHandle(actor, memory_capacity_bytes=15, worker_id="worker-0")

    running = handle.submit_tasks(
        [
            _FakeTask(
                name="running-standard",
                context={
                    "query_id": "query-execution-cap",
                    "node_id": "8",
                    "task_execution_class": "STANDARD",
                },
                inputs={"8": {"kind": "scan_task", "data": b"p0"}},
                plan={"plan": "exchange-template"},
            )
        ]
    )[0]
    first_waiting_for_node = handle.submit_tasks(
        [
            _FakeTask(
                name="node-wait-standard",
                context={
                    "query_id": "query-execution-cap",
                    "node_id": "9",
                    "task_execution_class": "STANDARD",
                },
                inputs={"9": {"kind": "scan_task", "data": b"p1"}},
                plan={"plan": "exchange-template"},
            )
        ]
    )
    ready_queued = handle.submit_tasks(
        [
            _FakeTask(
                name="ready-queued-standard",
                context={
                    "query_id": "query-execution-cap",
                    "node_id": "10",
                    "task_execution_class": "STANDARD",
                },
                inputs={"10": {"kind": "scan_task", "data": b"p2"}},
                plan={"plan": "exchange-template"},
            )
        ]
    )
    execution_deferred = handle.submit_tasks(
        [
            _FakeTask(
                name="execution-deferred-standard",
                context={
                    "query_id": "query-execution-cap",
                    "node_id": "11",
                    "task_execution_class": "STANDARD",
                },
                inputs={"11": {"kind": "scan_task", "data": b"p3"}},
                plan={"plan": "exchange-template"},
            )
        ]
    )

    assert isinstance(running, _FakeFteTaskHandle)
    assert first_waiting_for_node == []
    assert ready_queued == []
    assert execution_deferred == []
    assert [str(task_handle.task_id) for task_handle in handle.pop_fte_result_handles("query-execution-cap")] == [
        "query-execution-cap.0.0.0"
    ]
    stages_by_source = {
        next(iter(stage.dynamic_scan_source_node_ids)): stage
        for (query_id, _), stage in worker_handle_mod._FTE_FRAGMENT_EXECUTIONS.items()
        if query_id == "query-execution-cap"
    }
    node_wait_stage = stages_by_source["9"]
    ready_stage = stages_by_source["10"]
    deferred_stage = stages_by_source["11"]
    assert node_wait_stage.partitions[0].node_wait_started_at is not None
    assert ready_stage.partitions[0].ready_for_scheduling is True
    assert ready_stage.partitions[0].node_wait_started_at is not None
    assert deferred_stage.partitions[0].ready_for_scheduling is True
    assert deferred_stage.partitions[0].node_wait_started_at is not None
    assert deferred_stage.partitions[0].execution_ready_deferred is False
    assert ready_stage.waiting_for_execution_count() == 0
    assert handle.pop_fte_result_handles("query-execution-cap") == []
    assert deferred_stage.partitions[0].execution_ready_deferred is False

    handle.record_fte_task_terminal(running.task_id)
    scheduled = handle.pop_fte_result_handles("query-execution-cap")

    assert len(scheduled) == 1
    assert scheduled[0].task_id.query_id == "query-execution-cap"
    assert (
        sum(
            stage.partitions[0].node_wait_started_at is not None
            for stage in (node_wait_stage, ready_stage, deferred_stage)
        )
        == 2
    )
    assert deferred_stage.partitions[0].execution_ready_deferred is False
    assert len([call for call in actor.fte_calls if call[0] == "create"]) == 2


def test_fte_pending_drain_is_fair_across_queries(monkeypatch):
    monkeypatch.setattr(
        RayWorkerActorHandle,
        "_fte_task_handle_cls",
        staticmethod(lambda: _FakeFteTaskHandle),
    )
    monkeypatch.setattr(
        fragment_submission_mod,
        "_split_exchange_source_task_by_partition",
        lambda value: ([(int(value[-1:]), value)], 3, 3, False),
    )
    actor = _FakeActor()
    handle = RayWorkerActorHandle(actor, memory_capacity_bytes=15, worker_id="worker-0")

    query_a_handles = handle.submit_tasks(
        [
            _FakeTask(
                name="query-a-task-0",
                context={"query_id": "query-a", "node_id": "8"},
                inputs={"3": {"kind": "exchange_source_task", "data": b"p0"}},
                plan={"plan": "exchange-template"},
            ),
            _FakeTask(
                name="query-a-task-1",
                context={"query_id": "query-a", "node_id": "8"},
                inputs={"3": {"kind": "exchange_source_task", "data": b"p1"}},
                plan={"plan": "exchange-template"},
            ),
        ]
    )
    query_b_handles = handle.submit_tasks(
        [
            _FakeTask(
                name="query-b-task-0",
                context={"query_id": "query-b", "node_id": "8"},
                inputs={"3": {"kind": "exchange_source_task", "data": b"p0"}},
                plan={"plan": "exchange-template"},
            )
        ]
    )

    assert [str(task_handle.task_id) for task_handle in query_a_handles] == [
        "query-a.0.0.0",
    ]
    assert query_b_handles == []
    assert [str(task_handle.task_id) for task_handle in handle.pop_fte_result_handles("query-a")] == ["query-a.0.0.0"]

    handle.record_fte_task_terminal(query_a_handles[0].task_id)
    first_scheduled = handle.pop_fte_result_handles("query-b")

    assert [str(task_handle.task_id) for task_handle in first_scheduled] == ["query-b.0.0.0"]
    handle.record_fte_task_terminal(first_scheduled[0].task_id)
    second_scheduled = handle.pop_fte_result_handles("query-a")
    assert [str(task_handle.task_id) for task_handle in second_scheduled] == ["query-a.0.1.0"]


def test_fte_pending_drain_prefers_standard_over_speculative(monkeypatch):
    monkeypatch.setattr(
        RayWorkerActorHandle,
        "_fte_task_handle_cls",
        staticmethod(lambda: _FakeFteTaskHandle),
    )
    monkeypatch.setattr(
        fragment_submission_mod,
        "_split_exchange_source_task_by_partition",
        lambda value: ([(int(value[-1:]), value)], 2, 2, False),
    )
    actor = _FakeActor()
    handle = RayWorkerActorHandle(actor, memory_capacity_bytes=15, worker_id="worker-0")

    running = handle.submit_tasks(
        [
            _FakeTask(
                name="running-standard",
                context={
                    "query_id": "query-running-standard",
                    "node_id": "8",
                    "task_execution_class": "STANDARD",
                },
                inputs={"3": {"kind": "exchange_source_task", "data": b"p0"}},
                plan={"plan": "exchange-template"},
            )
        ]
    )[0]
    speculative_pending = handle.submit_tasks(
        [
            _FakeTask(
                name="pending-speculative",
                context={
                    "query_id": "query-pending-speculative",
                    "node_id": "8",
                    "task_execution_class": "SPECULATIVE",
                },
                inputs={"3": {"kind": "exchange_source_task", "data": b"p0"}},
                plan={"plan": "exchange-template"},
            )
        ]
    )
    standard_pending = handle.submit_tasks(
        [
            _FakeTask(
                name="pending-standard",
                context={
                    "query_id": "query-pending-standard",
                    "node_id": "8",
                    "task_execution_class": "STANDARD",
                },
                inputs={"3": {"kind": "exchange_source_task", "data": b"p0"}},
                plan={"plan": "exchange-template"},
            )
        ]
    )

    assert isinstance(running, _FakeFteTaskHandle)
    assert speculative_pending == []
    assert standard_pending == []
    assert [str(task_handle.task_id) for task_handle in handle.pop_fte_result_handles("query-running-standard")] == [
        "query-running-standard.0.0.0"
    ]

    handle.record_fte_task_terminal(running.task_id)
    first_scheduled = handle.pop_fte_result_handles("query-pending-standard")

    assert [str(task_handle.task_id) for task_handle in first_scheduled] == ["query-pending-standard.0.0.0"]
    handle.record_fte_task_terminal(first_scheduled[0].task_id)
    second_scheduled = handle.pop_fte_result_handles("query-pending-speculative")
    assert [str(task_handle.task_id) for task_handle in second_scheduled] == ["query-pending-speculative.0.0.0"]


def test_fte_immediate_speculative_waits_behind_standard_pending(monkeypatch):
    monkeypatch.setattr(
        RayWorkerActorHandle,
        "_fte_task_handle_cls",
        staticmethod(lambda: _FakeFteTaskHandle),
    )
    monkeypatch.setattr(
        fragment_submission_mod,
        "_split_exchange_source_task_by_partition",
        lambda value: ([(0, value)], 2, 1, False),
    )
    actor0 = _FakeActor()
    handle0 = RayWorkerActorHandle(actor0, memory_capacity_bytes=15, worker_id="worker-0")

    running = handle0.submit_tasks(
        [
            _FakeTask(
                name="running-standard",
                context={
                    "query_id": "query-immediate-running",
                    "node_id": "8",
                    "task_execution_class": "STANDARD",
                },
                inputs={"3": {"kind": "exchange_source_task", "data": b"p0"}},
                plan={"plan": "exchange-template"},
            )
        ]
    )[0]
    standard_pending = handle0.submit_tasks(
        [
            _FakeTask(
                name="pending-standard",
                context={
                    "query_id": "query-immediate-standard-pending",
                    "node_id": "8",
                    "task_execution_class": "STANDARD",
                },
                inputs={"3": {"kind": "exchange_source_task", "data": b"p1"}},
                plan={"plan": "exchange-template"},
            )
        ]
    )
    actor1 = _FakeActor()
    handle1 = RayWorkerActorHandle(actor1, memory_capacity_bytes=15, worker_id="worker-1")
    speculative_pending = handle1.submit_tasks(
        [
            _FakeTask(
                name="auto-speculative",
                context={"query_id": "query-immediate-auto-speculative", "node_id": "8"},
                inputs={"3": {"kind": "exchange_source_task", "data": b"p2"}},
                plan={"plan": "exchange-template"},
            )
        ]
    )

    assert isinstance(running, _FakeFteTaskHandle)
    assert standard_pending == []
    assert speculative_pending == []
    assert [str(task_handle.task_id) for task_handle in handle0.pop_fte_result_handles("query-immediate-running")] == [
        "query-immediate-running.0.0.0"
    ]
    assert [call[1]["task_id"]["query_id"] for call in actor1.fte_calls if call[0] == "create"] == []

    handle0.record_fte_task_terminal(running.task_id)
    scheduled = handle1.pop_fte_result_handles("query-immediate-standard-pending") + handle1.pop_fte_result_handles(
        "query-immediate-auto-speculative"
    )

    scheduled_ids = [str(task_handle.task_id) for task_handle in scheduled]
    assert scheduled_ids[0] == "query-immediate-standard-pending.0.0.0"
    assert "query-immediate-auto-speculative.0.0.0" in scheduled_ids


def test_fte_standard_reservation_obeys_shared_hard_memory_capacity(monkeypatch):
    monkeypatch.setattr(
        RayWorkerActorHandle,
        "_fte_task_handle_cls",
        staticmethod(lambda: _FakeFteTaskHandle),
    )
    monkeypatch.setattr(
        fragment_submission_mod,
        "_split_exchange_source_task_by_partition",
        lambda value: ([(int(value[-1:]), value)], 2, 2, False),
    )
    actor = _FakeActor()
    handle = RayWorkerActorHandle(actor, memory_capacity_bytes=15, worker_id="worker-0")

    speculative = handle.submit_tasks(
        [
            _FakeTask(
                name="running-speculative",
                context={
                    "query_id": "query-running-speculative",
                    "node_id": "8",
                    "task_execution_class": "SPECULATIVE",
                },
                inputs={"3": {"kind": "exchange_source_task", "data": b"p0"}},
                plan={"plan": "exchange-template"},
            )
        ]
    )[0]
    standard = handle.submit_tasks(
        [
            _FakeTask(
                name="running-standard",
                context={
                    "query_id": "query-standard-over-speculative",
                    "node_id": "8",
                    "task_execution_class": "STANDARD",
                },
                inputs={"3": {"kind": "exchange_source_task", "data": b"p0"}},
                plan={"plan": "exchange-template"},
            )
        ]
    )

    assert isinstance(speculative, _FakeFteTaskHandle)
    assert standard == []
    stats = handle.fte_pressure_stats()
    assert stats["assigned_memory_bytes"] == 10
    assert stats["standard_memory_bytes"] == 0
    assert stats["speculative_memory_bytes"] == 10


def test_fte_eager_speculative_drains_before_standard_without_blocking_it(monkeypatch):
    monkeypatch.setattr(
        RayWorkerActorHandle,
        "_fte_task_handle_cls",
        staticmethod(lambda: _FakeFteTaskHandle),
    )
    monkeypatch.setattr(
        fragment_submission_mod,
        "_split_exchange_source_task_by_partition",
        lambda value: ([(int(value[-1:]), value)], 2, 2, False),
    )
    actor = _FakeActor()
    handle = RayWorkerActorHandle(actor, memory_capacity_bytes=15, worker_id="worker-0")

    running = handle.submit_tasks(
        [
            _FakeTask(
                name="running-standard",
                context={
                    "query_id": "query-eager-running",
                    "node_id": "8",
                    "task_execution_class": "STANDARD",
                },
                inputs={"3": {"kind": "exchange_source_task", "data": b"p0"}},
                plan={"plan": "exchange-template"},
            )
        ]
    )[0]
    eager_pending = handle.submit_tasks(
        [
            _FakeTask(
                name="pending-eager",
                context={
                    "query_id": "query-pending-eager",
                    "node_id": "8",
                    "task_execution_class": "EAGER_SPECULATIVE",
                },
                inputs={"3": {"kind": "exchange_source_task", "data": b"p0"}},
                plan={"plan": "exchange-template"},
            )
        ]
    )
    standard_pending = handle.submit_tasks(
        [
            _FakeTask(
                name="pending-standard",
                context={
                    "query_id": "query-pending-after-eager",
                    "node_id": "8",
                    "task_execution_class": "STANDARD",
                },
                inputs={"3": {"kind": "exchange_source_task", "data": b"p0"}},
                plan={"plan": "exchange-template"},
            )
        ]
    )

    assert isinstance(running, _FakeFteTaskHandle)
    assert eager_pending == []
    assert standard_pending == []
    assert [str(task_handle.task_id) for task_handle in handle.pop_fte_result_handles("query-eager-running")] == [
        "query-eager-running.0.0.0"
    ]

    handle.record_fte_task_terminal(running.task_id)
    scheduled = handle.pop_fte_result_handles("query-pending-eager")

    assert [str(task_handle.task_id) for task_handle in scheduled] == ["query-pending-eager.0.0.0"]
    assert handle.pop_fte_result_handles("query-pending-after-eager") == []
    stats = handle.fte_pressure_stats()
    assert stats["standard_memory_bytes"] == 0
    assert stats["eager_speculative_memory_bytes"] == 10

    handle.record_fte_task_terminal(scheduled[0].task_id)
    standard_scheduled = handle.pop_fte_result_handles("query-pending-after-eager")

    assert [str(task_handle.task_id) for task_handle in standard_scheduled] == ["query-pending-after-eager.0.0.0"]


def test_fte_eager_speculative_cannot_overcommit_ray_memory(monkeypatch):
    monkeypatch.setattr(
        RayWorkerActorHandle,
        "_fte_task_handle_cls",
        staticmethod(lambda: _FakeFteTaskHandle),
    )
    monkeypatch.setattr(
        fragment_submission_mod,
        "_split_exchange_source_task_by_partition",
        lambda value: ([(int(value[-1:]), value)], 2, 2, False),
    )
    actor = _FakeActor()
    handle = RayWorkerActorHandle(actor, memory_capacity_bytes=15, worker_id="worker-0")
    standard = handle.submit_tasks(
        [
            _FakeTask(
                name="running-standard",
                context={
                    "query_id": "query-eager-overcommit-standard",
                    "node_id": "8",
                    "task_execution_class": "STANDARD",
                },
                inputs={"3": {"kind": "exchange_source_task", "data": b"p0"}},
                plan={"plan": "exchange-template"},
            )
        ]
    )[0]
    eager = handle.submit_tasks(
        [
            _FakeTask(
                name="running-eager",
                context={
                    "query_id": "query-eager-overcommit",
                    "node_id": "8",
                    "task_execution_class": "EAGER_SPECULATIVE",
                },
                inputs={"3": {"kind": "exchange_source_task", "data": b"p0"}},
                plan={"plan": "exchange-template"},
            )
        ]
    )

    assert isinstance(standard, _FakeFteTaskHandle)
    assert eager == []
    stats = handle.fte_pressure_stats()
    assert stats["assigned_memory_bytes"] == 10
    assert stats["standard_memory_bytes"] == 10
    assert stats["eager_speculative_memory_bytes"] == 0


def test_fte_speculative_cannot_overcommit_ray_memory(monkeypatch):
    monkeypatch.setattr(
        RayWorkerActorHandle,
        "_fte_task_handle_cls",
        staticmethod(lambda: _FakeFteTaskHandle),
    )
    monkeypatch.setattr(
        fragment_submission_mod,
        "_split_exchange_source_task_by_partition",
        lambda value: ([(int(value[-1:]), value)], 2, 2, False),
    )
    actor = _FakeActor()
    handle = RayWorkerActorHandle(actor, memory_capacity_bytes=15, worker_id="worker-0")
    standard = handle.submit_tasks(
        [
            _FakeTask(
                name="running-standard",
                context={
                    "query_id": "query-spec-overcommit-standard",
                    "node_id": "8",
                    "task_execution_class": "STANDARD",
                },
                inputs={"3": {"kind": "exchange_source_task", "data": b"p0"}},
                plan={"plan": "exchange-template"},
            )
        ]
    )[0]
    speculative = handle.submit_tasks(
        [
            _FakeTask(
                name="pending-speculative",
                context={
                    "query_id": "query-spec-overcommit",
                    "node_id": "8",
                    "task_execution_class": "SPECULATIVE",
                },
                inputs={"3": {"kind": "exchange_source_task", "data": b"p0"}},
                plan={"plan": "exchange-template"},
            )
        ]
    )

    assert isinstance(standard, _FakeFteTaskHandle)
    assert speculative == []
    assert [
        str(task_handle.task_id) for task_handle in handle.pop_fte_result_handles("query-spec-overcommit-standard")
    ] == ["query-spec-overcommit-standard.0.0.0"]


def test_fte_pending_execution_class_transition_does_not_bypass_hard_capacity(monkeypatch):
    monkeypatch.setattr(
        RayWorkerActorHandle,
        "_fte_task_handle_cls",
        staticmethod(lambda: _FakeFteTaskHandle),
    )
    monkeypatch.setattr(
        fragment_submission_mod,
        "_split_exchange_source_task_by_partition",
        lambda value: ([(int(value[-1:]), value)], 2, 2, False),
    )
    actor = _FakeActor()
    handle = RayWorkerActorHandle(actor, memory_capacity_bytes=15, worker_id="worker-0")

    running = handle.submit_tasks(
        [
            _FakeTask(
                name="running-speculative",
                context={
                    "query_id": "query-transition-blocker",
                    "node_id": "8",
                    "task_execution_class": "SPECULATIVE",
                },
                inputs={"3": {"kind": "exchange_source_task", "data": b"p0"}},
                plan={"plan": "exchange-template"},
            )
        ]
    )[0]
    pending = handle.submit_tasks(
        [
            _FakeTask(
                name="pending-speculative",
                context={
                    "query_id": "query-transition-pending",
                    "node_id": "8",
                    "task_execution_class": "SPECULATIVE",
                },
                inputs={"3": {"kind": "exchange_source_task", "data": b"p0"}},
                plan={"plan": "exchange-template"},
            )
        ]
    )

    assert isinstance(running, _FakeFteTaskHandle)
    assert pending == []
    assert [str(task_handle.task_id) for task_handle in handle.pop_fte_result_handles("query-transition-blocker")] == [
        "query-transition-blocker.0.0.0"
    ]
    scheduled = handle.set_fte_fragment_execution_execution_class(
        "query-transition-pending",
        "query-transition-pending:node:8",
        "STANDARD",
    )

    assert scheduled == []
    create_requests = [call[1] for call in actor.fte_calls if call[0] == "create"]
    assert [request["execution_class"] for request in create_requests] == ["SPECULATIVE"]
    stats = handle.fte_pressure_stats()
    assert stats["assigned_memory_bytes"] == 10
    assert stats["standard_memory_bytes"] == 0
    assert stats["speculative_memory_bytes"] == 10

    handle.record_fte_task_terminal(running.task_id)
    scheduled = handle.pop_fte_result_handles("query-transition-pending")

    assert [str(task_handle.task_id) for task_handle in scheduled] == ["query-transition-pending.0.0.0"]


def test_fte_running_execution_class_transition_updates_pressure(monkeypatch):
    monkeypatch.setattr(
        RayWorkerActorHandle,
        "_fte_task_handle_cls",
        staticmethod(lambda: _FakeFteTaskHandle),
    )
    actor = _FakeActor()
    handle = RayWorkerActorHandle(actor, memory_capacity_bytes=15, worker_id="worker-0")
    running = handle.submit_tasks(
        [
            _FakeTask(
                name="running-eager",
                context={
                    "query_id": "query-eager-transition",
                    "node_id": "7",
                    "task_execution_class": "EAGER_SPECULATIVE",
                },
                inputs={"7": {"kind": "scan_task", "data": b"a"}},
                plan={"plan": "scan-template"},
            )
        ]
    )[0]

    assert isinstance(running, _FakeFteTaskHandle)
    assert handle.fte_pressure_stats()["eager_speculative_memory_bytes"] == 10
    scheduled = handle.set_fte_fragment_execution_execution_class(
        "query-eager-transition",
        "query-eager-transition:node:7",
        "STANDARD",
    )

    assert scheduled == []
    stats = handle.fte_pressure_stats()
    assert stats["standard_memory_bytes"] == 10
    assert stats["eager_speculative_memory_bytes"] == 0
    with pytest.raises(ValueError):
        handle.set_fte_fragment_execution_execution_class(
            "query-eager-transition",
            "query-eager-transition:node:7",
            "SPECULATIVE",
        )


def test_fte_reservation_execution_class_transition_updates_pressure(monkeypatch):
    actor = _FakeActor()
    handle = RayWorkerActorHandle(actor, memory_capacity_bytes=15, worker_id="worker-0")
    handle.reserve_fte_partition(
        "query-reservation-transition",
        "fragment",
        0,
        memory_requirement_bytes=10,
        execution_class="SPECULATIVE",
    )

    assert handle.fte_pressure_stats()["speculative_memory_bytes"] == 10
    changed = handle.set_fte_partition_reservation_execution_class(
        "query-reservation-transition",
        "fragment",
        0,
        "STANDARD",
    )

    assert changed is True
    stats = handle.fte_pressure_stats()
    assert stats["standard_memory_bytes"] == 10
    assert stats["speculative_memory_bytes"] == 0
    with pytest.raises(ValueError):
        handle.set_fte_partition_reservation_execution_class(
            "query-reservation-transition",
            "fragment",
            0,
            "SPECULATIVE",
        )


def test_fte_worker_failure_retry_preserves_registered_heap(monkeypatch):
    monkeypatch.setattr(
        RayWorkerActorHandle,
        "_fte_task_handle_cls",
        staticmethod(lambda: _FakeFteTaskHandle),
    )
    actor0 = _FakeActor()
    actor1 = _FakeActor()
    actor2 = _FakeActor()
    failed_worker = RayWorkerActorHandle(actor0, memory_capacity_bytes=25, worker_id="worker-0")
    high_memory = RayWorkerActorHandle(actor1, memory_capacity_bytes=25, worker_id="worker-1")
    low_memory = RayWorkerActorHandle(actor2, memory_capacity_bytes=25, worker_id="worker-2")
    task = _FakeTask(
        name="scan-task",
        context={"query_id": "query-memory-retry", "node_id": "7"},
        inputs={"7": {"kind": "scan_task", "data": b"a"}},
        plan={"plan": "scan-template"},
    )
    first = failed_worker.submit_tasks([task])
    high_memory.reserve_fte_partition(
        "query-other",
        "fragment",
        0,
        memory_requirement_bytes=15,
    )
    low_memory.reserve_fte_partition(
        "query-other",
        "fragment",
        1,
        memory_requirement_bytes=5,
    )

    retries = low_memory.mark_fte_worker_failed("worker-0", "worker lost")

    assert len(first) == 1
    assert len(retries) == 1
    assert retries[0].worker_handle is low_memory
    assert worker_handle_mod._FTE_PARTITION_OWNERS[("query-memory-retry", "query-memory-retry:node:7", 0)] is low_memory
    retry_creates = [call for call in actor2.fte_calls if call[0] == "create"]
    assert len(retry_creates) == 1
    assert retry_creates[0][1]["memory_requirement_bytes"] == 10
    assert low_memory.fte_pressure_stats()["assigned_memory_bytes"] == 10
    assert high_memory.fte_pressure_stats()["assigned_memory_bytes"] == 0


def test_fte_worker_failure_replays_descriptor_on_new_owner(monkeypatch):
    monkeypatch.setattr(
        RayWorkerActorHandle,
        "_fte_task_handle_cls",
        staticmethod(lambda: _FakeFteTaskHandle),
    )
    actor0 = _FakeActor()
    actor1 = _FakeActor()
    handle0 = RayWorkerActorHandle(actor0, memory_capacity_bytes=1 << 60, worker_id="worker-0")
    handle1 = RayWorkerActorHandle(actor1, memory_capacity_bytes=1 << 60, worker_id="worker-1")
    task = _FakeTask(
        name="scan-task",
        context={"query_id": "query-fte-worker-lost", "node_id": "7"},
        inputs={"7": {"kind": "scan_task", "data": b"a"}},
        plan={"plan": "scan-template"},
    )

    first = handle0.submit_tasks([task])
    retries = handle1.mark_fte_worker_failed("worker-0", "actor died")

    assert len(first) == 1
    assert len(retries) == 1
    assert retries[0].worker_handle is handle1
    retry_creates = [call for call in actor1.fte_calls if call[0] == "create"]
    assert len(retry_creates) == 1
    assert actor1.register_payloads == [
        [
            {
                "fragment_id": "query-fte-worker-lost:node:7",
                "plan": {"plan": "scan-template"},
                "query_id": "query-fte-worker-lost",
            }
        ]
    ]
    retry_request = retry_creates[0][1]
    assert retry_request["task_id"]["attempt_id"] == 1
    assert retry_request["fragment_plan"] is None
    assert retry_request["initial_splits"]["7"][0]["data"] == b"a"
    assert "worker-0" not in worker_handle_mod._FTE_WORKER_HANDLES
    stats = handle1.fte_registry_stats()["event_schedulers"]["query-fte-worker-lost"]
    assert stats["event_counts"] == {
        "SplitEventsSubmitted": 1,
        "WorkerReservationCompleted": 2,
        "WorkerFailed": 1,
    }


def test_fte_worker_failure_keeps_retryability_partition_local():
    query_id = "query-fte-worker-lost-locality"
    fragment_id = f"{query_id}:node:7"
    actor0 = _FakeActor()
    actor1 = _FakeActor()
    failed = RayWorkerActorHandle(actor0, memory_capacity_bytes=1 << 60, worker_id="failed#0")
    replacement = RayWorkerActorHandle(actor1, memory_capacity_bytes=1 << 60, worker_id="replacement#0")
    stage = failed._get_or_create_fte_fragment_execution(
        {
            "query_id": query_id,
            "fragment_id": fragment_id,
            "cfg": {"cfg": "scan"},
            "context": {},
            "task_context_info": {},
        },
        dynamic_scan_sources={"7"},
        dynamic_exchange_sources=set(),
    )
    retryable_partition = stage.add_partition(
        0,
        NodeRequirements(host="replacement", remotely_accessible=False),
    )
    non_retryable_partition = stage.add_partition(
        1,
        NodeRequirements(host="missing", remotely_accessible=False),
    )
    retryable_partition.start_attempt(worker_id="failed#0", remote_handle=failed)
    non_retryable_partition.start_attempt(worker_id="failed#0", remote_handle=failed)
    worker_handle_mod._FTE_PARTITION_OWNERS[(query_id, fragment_id, 0)] = failed
    worker_handle_mod._FTE_PARTITION_OWNERS[(query_id, fragment_id, 1)] = failed

    handles = replacement.mark_fte_worker_failed("failed#0", "actor died")

    assert stage.partitions[0].failed is False
    assert stage.partitions[1].failed is True
    assert stage.failed is True
    assert all(str(handle.task_id) != f"{query_id}.0.1.1" for handle in handles)


def test_fte_worker_failure_retry_waits_for_scheduling_delayer(monkeypatch):
    monkeypatch.setenv("VANE_FTE_RETRY_INITIAL_DELAY_S", "0.5")
    monkeypatch.setenv("VANE_FTE_RETRY_MAX_DELAY_S", "2")
    monkeypatch.setenv("VANE_FTE_RETRY_DELAY_SCALE_FACTOR", "2")
    now = [100.0]
    monkeypatch.setattr(worker_handle_mod.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(
        RayWorkerActorHandle,
        "_fte_task_handle_cls",
        staticmethod(lambda: _FakeFteTaskHandle),
    )
    actor0 = _FakeActor()
    actor1 = _FakeActor()
    handle0 = RayWorkerActorHandle(actor0, memory_capacity_bytes=1 << 60, worker_id="worker-0")
    handle1 = RayWorkerActorHandle(actor1, memory_capacity_bytes=1 << 60, worker_id="worker-1")
    task = _FakeTask(
        name="scan-task",
        context={"query_id": "query-fte-retry-delay", "node_id": "7"},
        inputs={"7": {"kind": "scan_task", "data": b"a"}},
        plan={"plan": "scan-template"},
    )

    first = handle0.submit_tasks([task])

    assert len(first) == 1
    assert [str(task_handle.task_id) for task_handle in handle0.pop_fte_result_handles("query-fte-retry-delay")] == [
        "query-fte-retry-delay.0.0.0"
    ]

    retries = handle1.mark_fte_worker_failed("worker-0", "actor died")

    assert retries == []
    assert [call for call in actor1.fte_calls if call[0] == "create"] == []
    assert handle1.pop_fte_result_handles("query-fte-retry-delay") == []

    now[0] += 0.5
    scheduler = worker_handle_mod._FTE_SCHEDULERS.get("query-fte-retry-delay")
    assert scheduler is not None
    scheduler.enqueue(
        worker_handle_mod.RetryDelayExpired(
            "query-fte-retry-delay",
            scheduler.retry_delay_generation(),
        )
    )
    scheduled = scheduler.drain()

    assert len(scheduled) == 1
    assert scheduled[0].worker_handle is handle1
    assert str(scheduled[0].task_id) == "query-fte-retry-delay.0.0.1"
    retry_creates = [call for call in actor1.fte_calls if call[0] == "create"]
    assert len(retry_creates) == 1
    assert retry_creates[0][1]["task_id"]["attempt_id"] == 1
    assert retry_creates[0][1]["initial_splits"]["7"][0]["data"] == b"a"
    stats = handle1.fte_registry_stats()["event_schedulers"]["query-fte-retry-delay"]
    assert stats["event_counts"] == {
        "SplitEventsSubmitted": 1,
        "WorkerReservationCompleted": 2,
        "WorkerFailed": 1,
        "RetryDelayExpired": 1,
    }


def test_fte_split_append_control_failure_replays_on_replacement(monkeypatch):
    monkeypatch.setattr(
        RayWorkerActorHandle,
        "_fte_task_handle_cls",
        staticmethod(lambda: _FakeFteTaskHandle),
    )

    class _DeadOnSplitAppendActor(_FakeActor):
        def _fte_add_splits(self, task_id, source_node_id, splits):
            self.fte_calls.append(("add_splits", task_id, source_node_id, splits))
            raise RuntimeError("actor died during split append")

    actor0 = _DeadOnSplitAppendActor()
    actor1 = _FakeActor()
    handle0 = RayWorkerActorHandle(actor0, memory_capacity_bytes=1 << 60, worker_id="worker-0")
    handle1 = RayWorkerActorHandle(actor1, memory_capacity_bytes=1 << 60, worker_id="worker-1")
    first_task = _FakeTask(
        name="scan-task-0",
        context={"query_id": "query-fte-append-lost", "node_id": "7"},
        inputs={"7": {"kind": "scan_task", "data": b"a"}},
        plan={"plan": "scan-template"},
    )
    append_task = _FakeTask(
        name="scan-task-1",
        context={"query_id": "query-fte-append-lost", "node_id": "7"},
        inputs={"7": {"kind": "scan_task", "data": b"b"}},
        plan={"plan": "scan-template"},
    )

    first = handle0.submit_tasks([first_task])
    retries = handle0.submit_tasks([append_task])

    assert len(first) == 1
    assert first[0].worker_handle is handle0
    assert len(retries) == 1
    assert retries[0].worker_handle is handle1
    assert "worker-0" not in worker_handle_mod._FTE_WORKER_HANDLES
    assert (
        worker_handle_mod._FTE_PARTITION_OWNERS[("query-fte-append-lost", "query-fte-append-lost:node:7", 0)] is handle1
    )
    assert retries[0].worker_handle is handle1
    retry_creates = [call for call in actor1.fte_calls if call[0] == "create"]
    assert len(retry_creates) == 1
    retry_request = retry_creates[0][1]
    assert retry_request["task_id"]["attempt_id"] == 1
    assert [split["data"] for split in retry_request["initial_splits"]["7"]] == [b"a", b"b"]
    assert handle0.fte_pressure_stats()["running_attempt_count"] == 0
    assert handle1.fte_pressure_stats()["running_attempt_count"] == 1
    stats = handle0.fte_registry_stats()["event_schedulers"]["query-fte-append-lost"]
    assert stats["event_counts"] == {
        "SplitEventsSubmitted": 2,
        "WorkerReservationCompleted": 2,
        "WorkerFailed": 1,
    }


def test_fte_split_queue_full_replays_descriptor_on_replacement(monkeypatch):
    monkeypatch.setattr(
        RayWorkerActorHandle,
        "_fte_task_handle_cls",
        staticmethod(lambda: _FakeFteTaskHandle),
    )

    class _FullSplitQueueActor(_FakeActor):
        def _fte_wait_split_queue_has_space(
            self,
            task_id,
            source_node_id=None,
            max_buffered_splits=None,
            timeout_s=None,
        ):
            self.fte_calls.append(
                (
                    "wait_split_queue",
                    task_id,
                    source_node_id,
                    max_buffered_splits,
                    timeout_s,
                )
            )
            return {"has_space": False, "buffered_splits": 1024}

    actor0 = _FullSplitQueueActor()
    actor1 = _FakeActor()
    handle0 = RayWorkerActorHandle(actor0, memory_capacity_bytes=1 << 60, worker_id="worker-0")
    handle1 = RayWorkerActorHandle(actor1, memory_capacity_bytes=1 << 60, worker_id="worker-1")
    first_task = _FakeTask(
        name="scan-task-0",
        context={"query_id": "query-fte-queue-full", "node_id": "7"},
        inputs={"7": {"kind": "scan_task", "data": b"a"}},
        plan={"plan": "scan-template"},
    )
    append_task = _FakeTask(
        name="scan-task-1",
        context={"query_id": "query-fte-queue-full", "node_id": "7"},
        inputs={"7": {"kind": "scan_task", "data": b"b"}},
        plan={"plan": "scan-template"},
    )

    first = handle0.submit_tasks([first_task])
    retries = handle0.submit_tasks([append_task])

    assert len(first) == 1
    assert first[0].worker_handle is handle0
    assert len(retries) == 1
    assert retries[0].worker_handle is handle1
    assert [call[0] for call in actor0.fte_calls] == [
        "create",
        "wait_split_queue",
    ]
    retry_creates = [call for call in actor1.fte_calls if call[0] == "create"]
    assert len(retry_creates) == 1
    retry_request = retry_creates[0][1]
    assert retry_request["task_id"]["attempt_id"] == 1
    assert [split["data"] for split in retry_request["initial_splits"]["7"]] == [b"a", b"b"]
    assert "worker-0" not in worker_handle_mod._FTE_WORKER_HANDLES
    assert (
        worker_handle_mod._FTE_PARTITION_OWNERS[("query-fte-queue-full", "query-fte-queue-full:node:7", 0)] is handle1
    )
    stats = handle0.fte_registry_stats()["event_schedulers"]["query-fte-queue-full"]
    assert stats["event_counts"] == {
        "SplitEventsSubmitted": 2,
        "WorkerReservationCompleted": 2,
        "WorkerFailed": 1,
    }


def test_fte_worker_failure_replays_all_owned_stage_partitions(monkeypatch):
    monkeypatch.setattr(
        RayWorkerActorHandle,
        "_fte_task_handle_cls",
        staticmethod(lambda: _FakeFteTaskHandle),
    )
    actor0 = _FakeActor()
    handle0 = RayWorkerActorHandle(actor0, memory_capacity_bytes=1 << 60, worker_id="worker-0")

    scan_task = _FakeTask(
        name="scan-stage",
        context={"query_id": "query-host-loss", "node_id": "scan"},
        inputs={"7": {"kind": "scan_task", "data": b"scan-a"}},
        plan={"plan": "scan-template"},
    )
    downstream_descriptor = vane.ray_cxx.make_exchange_source_task_descriptor_for_test(
        [
            {
                "partition_id": 0,
                "attempt_id": 1,
                "node_id": "upstream-worker",
                "flight_port": 5010,
                "files": [
                    {
                        "path": "shuffle_query__sink_0__attempt_1/partition_0.arrow",
                        "file_size": 11,
                    }
                ],
            }
        ],
        [0],
        1,
        1,
    )
    exchange_task = _FakeTask(
        name="exchange-stage",
        context={
            "query_id": "query-host-loss",
            "node_id": "exchange",
            "task_execution_class": "STANDARD",
        },
        inputs={"3": {"kind": "exchange_source_task", "data": downstream_descriptor}},
        plan={"plan": "exchange-template"},
    )

    first_handles = handle0.submit_tasks([scan_task, exchange_task])
    actor1 = _FakeActor()
    handle1 = RayWorkerActorHandle(actor1, memory_capacity_bytes=1 << 60, worker_id="worker-1")

    retries = handle1.mark_fte_worker_failed("worker-0", "host lost")

    assert len(first_handles) == 2
    assert len(retries) == 2
    assert {str(handle.task_id) for handle in retries} == {
        "query-host-loss.0.0.1",
        "query-host-loss.1.0.1",
    }
    assert all(handle.worker_handle is handle1 for handle in retries)
    retry_creates = [call for call in actor1.fte_calls if call[0] == "create"]
    assert len(retry_creates) == 2
    retry_by_fragment = {request["fragment_id"]: request for _, request in retry_creates}

    scan_retry = retry_by_fragment["query-host-loss:node:scan"]
    assert scan_retry["task_id"]["attempt_id"] == 1
    assert scan_retry["initial_splits"]["7"][0]["data"] == b"scan-a"
    assert scan_retry["fragment_plan"] is None

    exchange_retry = retry_by_fragment["query-host-loss:node:exchange"]
    assert exchange_retry["task_id"]["attempt_id"] == 1
    assert exchange_retry["dynamic_exchange_source_node_ids"] == ["3"]
    source_handles = vane.ray_cxx.exchange_source_task_source_handles_for_test(
        exchange_retry["initial_splits"]["3"][0]["data"]
    )
    assert source_handles[0]["attempt_id"] == 1
    assert "__attempt_0" not in source_handles[0]["files"][0]["path"]

    assert actor1.register_payloads == [
        [
            {
                "fragment_id": "query-host-loss:node:scan",
                "plan": {"plan": "scan-template"},
                "query_id": "query-host-loss",
            }
        ],
        [
            {
                "fragment_id": "query-host-loss:node:exchange",
                "plan": {"plan": "exchange-template"},
                "query_id": "query-host-loss",
            }
        ],
    ]
    assert "worker-0" not in worker_handle_mod._FTE_WORKER_HANDLES
    assert worker_handle_mod._FTE_PARTITION_OWNERS[("query-host-loss", "query-host-loss:node:scan", 0)] is handle1
    assert worker_handle_mod._FTE_PARTITION_OWNERS[("query-host-loss", "query-host-loss:node:exchange", 0)] is handle1
    assert handle0.fte_pressure_stats()["running_attempt_count"] == 0
    assert handle1.fte_pressure_stats()["running_attempt_count"] == 2
    assert handle1.fte_pressure_stats()["reserved_partition_count"] == 0


def test_fte_host_loss_marks_sibling_workers_failed(monkeypatch):
    monkeypatch.setenv("VANE_FTE_CHAOS_FAIL_HOST_ON_WORKER_LOSS", "1")
    monkeypatch.setenv("VANE_FTE_CHAOS_KILL_WORKER_INDEX", "0,1")
    monkeypatch.setattr(
        RayWorkerActorHandle,
        "_fte_task_handle_cls",
        staticmethod(lambda: _FakeFteTaskHandle),
    )
    monkeypatch.setattr(
        fragment_submission_mod,
        "_split_exchange_source_task_by_partition",
        lambda value: ([(int(value[-1:]), value)], 2, 2, False),
    )
    killed = []
    monkeypatch.setattr(worker_handle_mod.ray, "kill", lambda actor: killed.append(actor))

    actor0 = _FakeActor()
    actor1 = _FakeActor()
    actor2 = _FakeActor()
    handle0 = RayWorkerActorHandle(actor0, memory_capacity_bytes=1 << 60, worker_id="host-a#0")
    handle1 = RayWorkerActorHandle(actor1, memory_capacity_bytes=1 << 60, worker_id="host-a#1")
    handle2 = RayWorkerActorHandle(actor2, memory_capacity_bytes=1 << 60, worker_id="host-a#2")
    task0 = _FakeTask(
        name="exchange-task-0",
        context={
            "query_id": "query-fte-host-chaos",
            "node_id": "8",
            "task_execution_class": "STANDARD",
        },
        inputs={"3": {"kind": "exchange_source_task", "data": b"p0"}},
        plan={"plan": "exchange-template"},
    )
    task1 = _FakeTask(
        name="exchange-task-1",
        context={
            "query_id": "query-fte-host-chaos",
            "node_id": "8",
            "task_execution_class": "STANDARD",
        },
        inputs={"3": {"kind": "exchange_source_task", "data": b"p1"}},
        plan={"plan": "exchange-template"},
    )

    first = handle0.submit_tasks([task0, task1])
    retries = handle2.mark_fte_worker_failed("host-a#0", "host lost")

    assert [handle.worker_handle for handle in first] == [handle0, handle1]
    assert len(retries) == 2
    assert {str(handle.task_id) for handle in retries} == {
        "query-fte-host-chaos.0.0.1",
        "query-fte-host-chaos.0.1.1",
    }
    assert all(handle.worker_handle is handle2 for handle in retries)
    assert "host-a#0" not in worker_handle_mod._FTE_WORKER_HANDLES
    assert "host-a#1" not in worker_handle_mod._FTE_WORKER_HANDLES
    assert worker_handle_mod._FTE_WORKER_HANDLES["host-a#2"] is handle2
    assert killed == [actor1]
    assert handle0.fte_pressure_stats()["running_attempt_count"] == 0
    assert handle1.fte_pressure_stats()["running_attempt_count"] == 0
    assert handle2.fte_pressure_stats()["running_attempt_count"] == 2


def test_fte_host_loss_failed_worker_set_is_stable_across_queries(monkeypatch):
    monkeypatch.setenv("VANE_FTE_CHAOS_FAIL_HOST_ON_WORKER_LOSS", "1")
    monkeypatch.setenv("VANE_FTE_CHAOS_KILL_WORKER_INDEX", "0,1")
    monkeypatch.setattr(
        RayWorkerActorHandle,
        "_fte_task_handle_cls",
        staticmethod(lambda: _FakeFteTaskHandle),
    )
    monkeypatch.setattr(
        fragment_submission_mod,
        "_split_exchange_source_task_by_partition",
        lambda value: ([(int(value[-1:]), value)], 2, 2, False),
    )
    monkeypatch.setattr(worker_handle_mod.ray, "kill", lambda _actor: None)

    actor0 = _FakeActor()
    actor1 = _FakeActor()
    handle0 = RayWorkerActorHandle(actor0, memory_capacity_bytes=1 << 60, worker_id="host-a#0")
    handle1 = RayWorkerActorHandle(actor1, memory_capacity_bytes=1 << 60, worker_id="host-a#1")

    for query_id in ("query-host-stable-a", "query-host-stable-b"):
        handle0.submit_tasks(
            [
                _FakeTask(
                    name=f"{query_id}-p0",
                    context={"query_id": query_id, "node_id": "8"},
                    inputs={"3": {"kind": "exchange_source_task", "data": b"p0"}},
                    plan={"plan": f"{query_id}-template"},
                ),
                _FakeTask(
                    name=f"{query_id}-p1",
                    context={"query_id": query_id, "node_id": "8"},
                    inputs={"3": {"kind": "exchange_source_task", "data": b"p1"}},
                    plan={"plan": f"{query_id}-template"},
                ),
            ]
        )

    actor2 = _FakeActor()
    handle2 = RayWorkerActorHandle(actor2, memory_capacity_bytes=1 << 60, worker_id="host-a#2")
    retries = handle2.mark_fte_worker_failed("host-a#0", "host lost")

    assert retries == []

    retries_after_eof = handle2.task_input_stream_exhausted(["3"])

    assert {str(handle.task_id) for handle in retries_after_eof} == {
        "query-host-stable-a.0.0.1",
        "query-host-stable-a.0.1.1",
        "query-host-stable-b.0.0.1",
        "query-host-stable-b.0.1.1",
    }
    assert all(handle.worker_handle is handle2 for handle in retries)
    assert handle0.fte_pressure_stats()["running_attempt_count"] == 0
    assert handle1.fte_pressure_stats()["running_attempt_count"] == 0
    assert handle2.fte_pressure_stats()["running_attempt_count"] == 4


def test_fte_worker_failure_without_replacement_fails_stage_without_retry(monkeypatch):
    monkeypatch.setattr(
        RayWorkerActorHandle,
        "_fte_task_handle_cls",
        staticmethod(lambda: _FakeFteTaskHandle),
    )
    actor = _FakeActor()
    handle = RayWorkerActorHandle(actor, memory_capacity_bytes=1 << 60, worker_id="worker-alone")
    task = _FakeTask(
        name="scan-task",
        context={"query_id": "query-no-replacement", "node_id": "7"},
        inputs={"7": {"kind": "scan_task", "data": b"a"}},
        plan={"plan": "scan-template"},
    )

    first = handle.submit_tasks([task])
    retries = handle.mark_fte_worker_failed("worker-alone", "host lost")

    assert len(first) == 1
    assert retries == []
    assert "worker-alone" not in worker_handle_mod._FTE_WORKER_HANDLES
    assert (
        "query-no-replacement",
        "query-no-replacement:node:7",
        0,
    ) not in worker_handle_mod._FTE_PARTITION_OWNERS
    stage = worker_handle_mod._FTE_FRAGMENT_EXECUTIONS[("query-no-replacement", "query-no-replacement:node:7")]
    partition = stage.partitions[0]
    assert stage.failed is True
    assert partition.failed is True
    assert partition.running_attempts == {}
    assert handle.fte_pressure_stats()["running_attempt_count"] == 0


def test_fte_exchange_tasks_create_one_task_per_partition(monkeypatch):
    monkeypatch.setattr(
        RayWorkerActorHandle,
        "_fte_task_handle_cls",
        staticmethod(lambda: _FakeFteTaskHandle),
    )
    monkeypatch.setattr(
        fragment_submission_mod,
        "_split_exchange_source_task_by_partition",
        lambda value: ([(int(value[-1:]), value)], 2, 2, False),
    )
    actor = _FakeActor()
    handle = RayWorkerActorHandle(actor, memory_capacity_bytes=1 << 60)
    task0 = _FakeTask(
        name="exchange-task-0",
        context={"query_id": "query-fte-exchange", "node_id": "8"},
        inputs={"3": {"kind": "exchange_source_task", "data": b"p0"}},
        plan={"plan": "exchange-template"},
    )
    task1 = _FakeTask(
        name="exchange-task-1",
        context={"query_id": "query-fte-exchange", "node_id": "8"},
        inputs={"3": {"kind": "exchange_source_task", "data": b"p1"}},
        plan={"plan": "exchange-template"},
    )

    handles = handle.submit_tasks([task0, task1])

    assert all(isinstance(handle, _FakeFteTaskHandle) for handle in handles)
    create_calls = [call for call in actor.fte_calls if call[0] == "create"]
    assert len(create_calls) == 2
    assert [call[1]["task_id"]["partition_id"] for call in create_calls] == [0, 1]
    assert [handle.task_id.partition_id for handle in handles] == [0, 1]
    for _, request in create_calls:
        assert "exchange_source_task:3" not in request["context"]
        assert "exchange_source_task_nodes" not in request["context"]
        assert request["dynamic_exchange_source_node_ids"] == ["3"]
    assert create_calls[0][1]["initial_splits"]["3"][0]["data"] == b"p0"
    assert create_calls[0][1]["initial_splits"]["3"][0].get("source_partition_id", 0) == 0
    assert create_calls[1][1]["initial_splits"]["3"][0]["data"] == b"p1"
    assert create_calls[1][1]["initial_splits"]["3"][0]["source_partition_id"] == 1


def test_fte_downstream_exchange_source_propagates_only_selected_retry_attempt(monkeypatch):
    monkeypatch.setattr(
        RayWorkerActorHandle,
        "_fte_task_handle_cls",
        staticmethod(lambda: _FakeFteTaskHandle),
    )
    selected_handles = [
        {
            "partition_id": 0,
            "attempt_id": 1,
            "node_id": "worker-retry",
            "flight_port": 5010,
            "files": [
                {
                    "path": "shuffle_query__sink_0__attempt_1/partition_0.arrow",
                    "file_size": 11,
                }
            ],
        },
        {
            "partition_id": 1,
            "attempt_id": 1,
            "node_id": "worker-retry",
            "flight_port": 5010,
            "files": [
                {
                    "path": "shuffle_query__sink_1__attempt_1/partition_1.arrow",
                    "file_size": 17,
                }
            ],
        },
    ]
    downstream_descriptor = vane.ray_cxx.make_exchange_source_task_descriptor_for_test(
        selected_handles,
        [0, 1],
        2,
        2,
    )
    actor = _FakeActor()
    handle = RayWorkerActorHandle(actor, memory_capacity_bytes=1 << 60, worker_id="downstream-worker")
    task = _FakeTask(
        name="downstream-aggregate",
        context={"query_id": "query-selected-downstream", "node_id": "9"},
        inputs={"3": {"kind": "exchange_source_task", "data": downstream_descriptor}},
        plan={"plan": "downstream-template"},
    )

    handles = handle.submit_tasks([task])

    assert all(isinstance(item, _FakeFteTaskHandle) for item in handles)
    create_calls = [call for call in actor.fte_calls if call[0] == "create"]
    assert len(create_calls) == 2
    assert [call[1]["task_id"]["partition_id"] for call in create_calls] == [0, 1]
    for idx, (_, request) in enumerate(create_calls):
        assert "exchange_source_task:3" not in request["context"]
        assert "exchange_source_task_nodes" not in request["context"]
        assert request["dynamic_exchange_source_node_ids"] == ["3"]
        split = request["initial_splits"]["3"][0]
        assert split["kind"] == "exchange_source_task"
        assert split["source_partition_id"] == idx
        source_handles = vane.ray_cxx.exchange_source_task_source_handles_for_test(split["data"])
        assert source_handles == [selected_handles[idx]]
        assert source_handles[0]["attempt_id"] == 1
        assert source_handles[0]["node_id"] == "worker-retry"
        assert "__attempt_0" not in source_handles[0]["files"][0]["path"]


def test_fte_exchange_source_task_count_merges_source_partitions(monkeypatch):
    monkeypatch.setattr(
        RayWorkerActorHandle,
        "_fte_task_handle_cls",
        staticmethod(lambda: _FakeFteTaskHandle),
    )
    actor = _FakeActor()
    handle = RayWorkerActorHandle(actor, memory_capacity_bytes=1 << 60)
    task = _FakeTask(
        name="exchange-task-merged",
        context={"query_id": "query-fte-exchange-merge", "node_id": "8"},
        inputs={
            "3": {
                "kind": "exchange_source_task",
                "data": {
                    "partition_indices": [0, 1],
                    "source_partition_count": 4,
                    "source_task_count": 2,
                },
            }
        },
        plan={"plan": "exchange-template"},
    )

    handles = handle.submit_tasks([task])

    assert len(handles) == 1
    create_calls = [call for call in actor.fte_calls if call[0] == "create"]
    add_calls = [call for call in actor.fte_calls if call[0] == "add_splits"]
    assert len(create_calls) == 1
    assert create_calls[0][1]["task_id"]["partition_id"] == 0
    assert add_calls == []
    assert [split["source_partition_id"] for split in create_calls[0][1]["initial_splits"]["3"]] == [0, 1]


def test_fte_exchange_source_descriptor_defers_no_more_until_stream_exhausted(monkeypatch):
    monkeypatch.setattr(
        RayWorkerActorHandle,
        "_fte_task_handle_cls",
        staticmethod(lambda: _FakeFteTaskHandle),
    )
    actor = _FakeActor()
    handle = RayWorkerActorHandle(actor, memory_capacity_bytes=1 << 60)
    task = _FakeTask(
        name="exchange-task-final",
        context={"query_id": "query-fte-exchange-final", "node_id": "8"},
        inputs={
            "3": {
                "kind": "exchange_source_task",
                "data": {
                    "partition_indices": [0, 1],
                    "source_partition_count": 2,
                    "source_task_count": 1,
                },
            }
        },
        plan={"plan": "exchange-template"},
    )

    handles = handle.submit_tasks([task])

    assert len(handles) == 1
    create_requests = [call[1] for call in actor.fte_calls if call[0] == "create"]
    assert len(create_requests) == 1
    assert create_requests[0]["no_more_splits"] == []
    assert [split["source_partition_id"] for split in create_requests[0]["initial_splits"]["3"]] == [0, 1]

    actor.fte_calls.clear()
    exhausted_handles = handle.task_input_stream_exhausted(["3"])

    assert exhausted_handles == []
    assert [call[0] for call in actor.fte_calls] == ["no_more_splits"]
    assert actor.fte_calls[0][2] == "3"


def test_fte_exchange_source_stream_exhaustion_sends_no_more_to_running_task(monkeypatch):
    monkeypatch.setattr(
        RayWorkerActorHandle,
        "_fte_task_handle_cls",
        staticmethod(lambda: _FakeFteTaskHandle),
    )
    actor = _FakeActor()
    handle = RayWorkerActorHandle(actor, memory_capacity_bytes=1 << 60)
    first_task = _FakeTask(
        name="exchange-task-final-running-0",
        context={"query_id": "query-fte-exchange-final-running", "node_id": "8"},
        inputs={
            "3": {
                "kind": "exchange_source_task",
                "data": {
                    "partition_indices": [0],
                    "source_partition_count": 2,
                    "source_task_count": 1,
                },
            }
        },
        plan={"plan": "exchange-template"},
    )
    second_task = _FakeTask(
        name="exchange-task-final-running-1",
        context={"query_id": "query-fte-exchange-final-running", "node_id": "8"},
        inputs={
            "3": {
                "kind": "exchange_source_task",
                "data": {
                    "partition_indices": [1],
                    "source_partition_count": 2,
                    "source_task_count": 1,
                },
            }
        },
        plan={"plan": "exchange-template"},
    )

    first_handles = handle.submit_tasks([first_task])
    actor.fte_calls.clear()
    second_handles = handle.submit_tasks([second_task])

    assert len(first_handles) == 1
    assert second_handles == []
    assert [call[0] for call in actor.fte_calls] == [
        "wait_split_queue",
        "add_splits",
    ]
    add_call = actor.fte_calls[1]
    assert add_call[2] == "3"
    assert [split["source_partition_id"] for split in add_call[3]] == [1]

    actor.fte_calls.clear()
    exhausted_handles = handle.task_input_stream_exhausted(["3"])

    assert exhausted_handles == []
    assert [call[0] for call in actor.fte_calls] == ["no_more_splits"]
    no_more_call = actor.fte_calls[0]
    assert no_more_call[2] == "3"


def test_fte_exchange_selector_event_updates_running_consumer(monkeypatch):
    monkeypatch.setattr(
        RayWorkerActorHandle,
        "_fte_task_handle_cls",
        staticmethod(lambda: _FakeFteTaskHandle),
    )
    actor = _FakeActor()
    handle = RayWorkerActorHandle(actor, memory_capacity_bytes=1 << 60)
    first_task = _FakeTask(
        name="exchange-selector-event-0",
        context={"query_id": "query-fte-selector-event", "node_id": "8"},
        inputs={
            "3": {
                "kind": "exchange_source_task",
                "data": {
                    "partition_indices": [0],
                    "source_partition_count": 2,
                    "source_task_count": 1,
                },
            }
        },
        plan={"plan": "exchange-template"},
    )

    first_handles = handle.submit_tasks([first_task])
    actor.fte_calls.clear()
    handles = handle.update_fte_exchange_selector(
        "query-fte-selector-event",
        "query-fte-selector-event:node:8",
        "3",
        selector=_exchange_selector_payload(
            [
                {
                    "source_node_id": "3",
                    "sequence_id": 1,
                    "kind": "exchange_source_task",
                    "data": {
                        "partition_indices": [1],
                        "source_partition_count": 2,
                        "source_task_count": 1,
                    },
                    "source_partition_id": 1,
                }
            ],
            final=True,
            partition_count=2,
            selected={"0": None},
        ),
    )

    assert len(first_handles) == 1
    assert handles == []
    assert [call[0] for call in actor.fte_calls] == [
        "wait_split_queue",
        "add_splits",
        "no_more_splits",
    ]
    assert [split["source_partition_id"] for split in actor.fte_calls[1][3]] == [1]
    stats = handle.fte_registry_stats()["event_schedulers"]["query-fte-selector-event"]
    assert stats["event_counts"] == {
        "SplitEventsSubmitted": 1,
        "WorkerReservationCompleted": 1,
        "ExchangeSelectorUpdated": 1,
    }


def test_fte_exchange_selector_event_requires_selector_payload(monkeypatch):
    monkeypatch.setattr(
        RayWorkerActorHandle,
        "_fte_task_handle_cls",
        staticmethod(lambda: _FakeFteTaskHandle),
    )
    actor = _FakeActor()
    handle = RayWorkerActorHandle(actor, memory_capacity_bytes=1 << 60)
    handle.submit_tasks(
        [
            _FakeTask(
                name="exchange-selector-missing-selector-0",
                context={"query_id": "query-fte-selector-required", "node_id": "8"},
                inputs={
                    "3": {
                        "kind": "exchange_source_task",
                        "data": {
                            "partition_indices": [0],
                            "source_partition_count": 1,
                            "source_task_count": 1,
                        },
                    }
                },
                plan={"plan": "exchange-template"},
            )
        ]
    )

    with pytest.raises(ValueError, match="requires selector payload"):
        handle.update_fte_exchange_selector(
            "query-fte-selector-required",
            "query-fte-selector-required:node:8",
            "3",
        )


def test_fte_exchange_selector_event_updates_running_and_pending_consumers(monkeypatch):
    _register_test_query_graph(
        "query-fte-selector-mixed",
        ["query-fte-selector-mixed:node:8"],
        max_concurrency=1,
    )
    monkeypatch.setattr(
        RayWorkerActorHandle,
        "_fte_task_handle_cls",
        staticmethod(lambda: _FakeFteTaskHandle),
    )
    actor = _FakeActor()
    handle = RayWorkerActorHandle(actor, memory_capacity_bytes=100)
    first_task = _FakeTask(
        name="exchange-selector-mixed-0",
        context={"query_id": "query-fte-selector-mixed", "node_id": "8"},
        inputs={
            "3": {
                "kind": "exchange_source_task",
                "data": {
                    "partition_indices": [0],
                    "source_partition_count": 2,
                    "source_task_count": 2,
                },
            }
        },
        plan={"plan": "exchange-template"},
    )

    first_handles = handle.submit_tasks([first_task])
    assert [str(task_handle.task_id) for task_handle in first_handles] == ["query-fte-selector-mixed.0.0.0"]
    stage = worker_handle_mod._FTE_FRAGMENT_EXECUTIONS[("query-fte-selector-mixed", "query-fte-selector-mixed:node:8")]
    assert sorted(stage.partitions) == [0, 1]
    assert stage.partitions[0].running_attempt is not None
    assert stage.partitions[1].running_attempt is None

    actor.fte_calls.clear()
    handles = handle.update_fte_exchange_selector(
        "query-fte-selector-mixed",
        "query-fte-selector-mixed:node:8",
        "3",
        selector=_exchange_selector_payload(
            [
                {
                    "source_node_id": "3",
                    "sequence_id": 1,
                    "kind": "exchange_source_task",
                    "data": {
                        "partition_indices": [0],
                        "source_partition_count": 2,
                        "source_task_count": 2,
                        "source_handles": [{"partition_id": 0, "attempt_id": 1, "path": "selected-0"}],
                    },
                    "source_partition_id": 0,
                },
                {
                    "source_node_id": "3",
                    "sequence_id": 2,
                    "kind": "exchange_source_task",
                    "data": {
                        "partition_indices": [1],
                        "source_partition_count": 2,
                        "source_task_count": 2,
                        "source_handles": [{"partition_id": 1, "attempt_id": 1, "path": "selected-1"}],
                    },
                    "source_partition_id": 1,
                },
            ],
            final=True,
            partition_count=2,
        ),
    )

    assert handles == []
    assert [call[0] for call in actor.fte_calls] == [
        "wait_split_queue",
        "add_splits",
        "no_more_splits",
    ]
    assert actor.fte_calls[1][1]["partition_id"] == 0
    assert [split["source_partition_id"] for split in actor.fte_calls[1][3]] == [0]
    assert actor.fte_calls[2][1]["partition_id"] == 0
    assert stage.partitions[1].running_attempt is None
    assert stage.partitions[1].ready_for_scheduling is False
    assert stage.partitions[1].execution_ready_deferred is True
    assert stage.partitions[1].descriptor.no_more_splits == {"3"}
    assert [split.source_partition_id for split in stage.partitions[1].descriptor.initial_splits["3"]] == [1]

    actor.fte_calls.clear()
    scheduled = handle.handle_fte_task_status(
        {
            "state": "FINISHED",
            "task_id": first_handles[0].task_id.to_dict(),
            "version": 1,
        }
    )

    assert [str(task_handle.task_id) for task_handle in scheduled] == ["query-fte-selector-mixed.0.1.0"]
    create_calls = [call for call in actor.fte_calls if call[0] == "create"]
    assert len(create_calls) == 1
    request = create_calls[0][1]
    assert request["task_id"]["partition_id"] == 1
    assert request["no_more_splits"] == ["3"]
    assert [split["source_partition_id"] for split in request["initial_splits"]["3"]] == [1]
    stats = handle.fte_registry_stats()["event_schedulers"]["query-fte-selector-mixed"]
    assert stats["event_counts"] == {
        "SplitEventsSubmitted": 1,
        "WorkerReservationCompleted": 2,
        "ExchangeSelectorUpdated": 1,
        "TaskStatusChanged": 1,
    }


def test_fte_exchange_selector_event_deduplicates_duplicate_source_handles(monkeypatch):
    monkeypatch.setattr(
        RayWorkerActorHandle,
        "_fte_task_handle_cls",
        staticmethod(lambda: _FakeFteTaskHandle),
    )
    actor = _FakeActor()
    handle = RayWorkerActorHandle(actor, memory_capacity_bytes=1 << 60)
    first_task = _FakeTask(
        name="exchange-selector-duplicate-0",
        context={"query_id": "query-fte-selector-duplicate", "node_id": "8"},
        inputs={
            "3": {
                "kind": "exchange_source_task",
                "data": {
                    "partition_indices": [0],
                    "source_partition_count": 2,
                    "source_task_count": 1,
                },
            }
        },
        plan={"plan": "downstream-template"},
    )

    first_handles = handle.submit_tasks([first_task])
    actor.fte_calls.clear()
    duplicate_split = {
        "source_node_id": "3",
        "kind": "exchange_source_task",
        "data": {
            "partition_indices": [1],
            "source_partition_count": 2,
            "source_task_count": 1,
            "source_handles": [{"partition_id": 1, "attempt_id": 1, "path": "selected"}],
        },
        "source_partition_id": 1,
    }
    handles = handle.update_fte_exchange_selector(
        "query-fte-selector-duplicate",
        "query-fte-selector-duplicate:node:8",
        "3",
        selector={
            "final": True,
            "partition_count": 2,
            "selected": {"0": None},
            "splits": [
                {**duplicate_split, "sequence_id": 1},
                {**duplicate_split, "sequence_id": 2},
            ],
        },
    )

    assert len(first_handles) == 1
    assert handles == []
    assert [call[0] for call in actor.fte_calls] == [
        "wait_split_queue",
        "add_splits",
        "no_more_splits",
    ]
    add_call = actor.fte_calls[1]
    assert add_call[2] == "3"
    assert [split["source_partition_id"] for split in add_call[3]] == [1]


def test_fte_exchange_selector_version_stale_update_is_ignored(monkeypatch):
    monkeypatch.setattr(
        RayWorkerActorHandle,
        "_fte_task_handle_cls",
        staticmethod(lambda: _FakeFteTaskHandle),
    )
    actor = _FakeActor()
    handle = RayWorkerActorHandle(actor, memory_capacity_bytes=1 << 60)
    first_task = _FakeTask(
        name="exchange-selector-versioned-0",
        context={"query_id": "query-fte-selector-versioned", "node_id": "8"},
        inputs={
            "3": {
                "kind": "exchange_source_task",
                "data": {
                    "partition_indices": [0],
                    "source_partition_count": 2,
                    "source_task_count": 1,
                },
            }
        },
        plan={"plan": "downstream-template"},
    )
    selected_split = {
        "source_node_id": "3",
        "sequence_id": 1,
        "kind": "exchange_source_task",
        "data": {
            "partition_indices": [1],
            "source_partition_count": 2,
            "source_task_count": 1,
            "source_handles": [{"partition_id": 1, "attempt_id": 1, "path": "selected"}],
        },
        "source_partition_id": 1,
    }

    first_handles = handle.submit_tasks([first_task])
    actor.fte_calls.clear()
    handles = handle.update_fte_exchange_selector(
        "query-fte-selector-versioned",
        "query-fte-selector-versioned:node:8",
        "3",
        selector={
            "version": 1,
            "partition_count": 2,
            "selected": {"1": {"attempt_id": 1, "split": selected_split}},
        },
    )

    assert len(first_handles) == 1
    assert handles == []
    assert [call[0] for call in actor.fte_calls] == ["wait_split_queue", "add_splits"]
    selector_stats = handle.fte_registry_stats()["queries"]["query-fte-selector-versioned"]["fragment_executions"][
        "query-fte-selector-versioned:node:8"
    ]["exchange_selectors"]["3"]
    assert selector_stats["version"] == 1
    assert selector_stats["final"] is False
    assert selector_stats["selected_partitions"] == [1]

    actor.fte_calls.clear()
    stale_handles = handle.update_fte_exchange_selector(
        "query-fte-selector-versioned",
        "query-fte-selector-versioned:node:8",
        "3",
        selector={
            "version": 0,
            "partition_count": 2,
            "selected": {"1": {"attempt_id": 1, "split": selected_split}},
        },
    )

    assert stale_handles == []
    assert actor.fte_calls == []


def test_fte_exchange_selector_materializes_preselected_partition(monkeypatch):
    monkeypatch.setattr(
        RayWorkerActorHandle,
        "_fte_task_handle_cls",
        staticmethod(lambda: _FakeFteTaskHandle),
    )
    actor = _FakeActor()
    handle = RayWorkerActorHandle(actor, memory_capacity_bytes=1 << 60)
    first_task = _FakeTask(
        name="exchange-selector-materialize-0",
        context={"query_id": "query-fte-selector-materialize", "node_id": "8"},
        inputs={
            "3": {
                "kind": "exchange_source_task",
                "data": {
                    "partition_indices": [0],
                    "source_partition_count": 2,
                    "source_task_count": 1,
                },
            }
        },
        plan={"plan": "downstream-template"},
    )
    selected_split = {
        "source_node_id": "3",
        "sequence_id": 1,
        "kind": "exchange_source_task",
        "data": {
            "partition_indices": [1],
            "source_partition_count": 2,
            "source_task_count": 1,
            "source_handles": [{"partition_id": 1, "attempt_id": 0, "path": "selected"}],
        },
        "source_partition_id": 1,
    }

    handle.submit_tasks([first_task])
    actor.fte_calls.clear()
    metadata_handles = handle.update_fte_exchange_selector(
        "query-fte-selector-materialize",
        "query-fte-selector-materialize:node:8",
        "3",
        selector={
            "version": 1,
            "partition_count": 2,
            "selected": {"1": {"attempt_id": 0}},
        },
    )

    assert metadata_handles == []
    assert actor.fte_calls == []

    materialized_handles = handle.update_fte_exchange_selector(
        "query-fte-selector-materialize",
        "query-fte-selector-materialize:node:8",
        "3",
        selector={
            "version": 2,
            "partition_count": 2,
            "selected": {"1": {"split": selected_split}},
        },
    )

    assert materialized_handles == []
    assert [call[0] for call in actor.fte_calls] == ["wait_split_queue", "add_splits"]
    add_call = actor.fte_calls[1]
    assert add_call[2] == "3"
    assert [split["source_partition_id"] for split in add_call[3]] == [1]
    selector_stats = handle.fte_registry_stats()["queries"]["query-fte-selector-materialize"]["fragment_executions"][
        "query-fte-selector-materialize:node:8"
    ]["exchange_selectors"]["3"]
    assert selector_stats["version"] == 2
    assert selector_stats["selected_attempts"]["1"] == 0


def test_fte_exchange_selector_version_rejects_conflicts(monkeypatch):
    monkeypatch.setattr(
        RayWorkerActorHandle,
        "_fte_task_handle_cls",
        staticmethod(lambda: _FakeFteTaskHandle),
    )
    actor = _FakeActor()
    handle = RayWorkerActorHandle(actor, memory_capacity_bytes=1 << 60)
    first_task = _FakeTask(
        name="exchange-selector-conflict-0",
        context={"query_id": "query-fte-selector-conflict", "node_id": "8"},
        inputs={
            "3": {
                "kind": "exchange_source_task",
                "data": {
                    "partition_indices": [0],
                    "source_partition_count": 2,
                    "source_task_count": 1,
                },
            }
        },
        plan={"plan": "downstream-template"},
    )
    selected_split = {
        "source_node_id": "3",
        "sequence_id": 1,
        "kind": "exchange_source_task",
        "data": {
            "partition_indices": [1],
            "source_partition_count": 2,
            "source_task_count": 1,
            "source_handles": [{"partition_id": 1, "attempt_id": 1, "path": "selected"}],
        },
        "source_partition_id": 1,
    }

    handle.submit_tasks([first_task])
    handle.update_fte_exchange_selector(
        "query-fte-selector-conflict",
        "query-fte-selector-conflict:node:8",
        "3",
        selector={
            "version": 1,
            "partition_count": 2,
            "selected": {"1": {"attempt_id": 1, "split": selected_split}},
        },
    )
    actor.fte_calls.clear()

    with pytest.raises(ValueError, match="conflicting exchange selector update"):
        handle.update_fte_exchange_selector(
            "query-fte-selector-conflict",
            "query-fte-selector-conflict:node:8",
            "3",
            selector={
                "version": 1,
                "partition_count": 2,
                "selected": {"1": {"attempt_id": 2, "split": selected_split}},
            },
        )

    with pytest.raises(ValueError, match="cannot change selected attempt"):
        handle.update_fte_exchange_selector(
            "query-fte-selector-conflict",
            "query-fte-selector-conflict:node:8",
            "3",
            selector={
                "version": 2,
                "partition_count": 2,
                "selected": {"1": {"attempt_id": 2, "split": selected_split}},
            },
        )
    assert actor.fte_calls == []


def test_fte_exchange_selector_final_requires_full_coverage(monkeypatch):
    monkeypatch.setattr(
        RayWorkerActorHandle,
        "_fte_task_handle_cls",
        staticmethod(lambda: _FakeFteTaskHandle),
    )
    actor = _FakeActor()
    handle = RayWorkerActorHandle(actor, memory_capacity_bytes=1 << 60)
    first_task = _FakeTask(
        name="exchange-selector-final-coverage-0",
        context={"query_id": "query-fte-selector-final-coverage", "node_id": "8"},
        inputs={
            "3": {
                "kind": "exchange_source_task",
                "data": {
                    "partition_indices": [0],
                    "source_partition_count": 2,
                    "source_task_count": 1,
                },
            }
        },
        plan={"plan": "downstream-template"},
    )
    selected_split = {
        "source_node_id": "3",
        "sequence_id": 1,
        "kind": "exchange_source_task",
        "data": {
            "partition_indices": [1],
            "source_partition_count": 2,
            "source_task_count": 1,
            "source_handles": [{"partition_id": 1, "attempt_id": 1, "path": "selected"}],
        },
        "source_partition_id": 1,
    }

    handle.submit_tasks([first_task])

    with pytest.raises(ValueError, match="missing partitions"):
        handle.update_fte_exchange_selector(
            "query-fte-selector-final-coverage",
            "query-fte-selector-final-coverage:node:8",
            "3",
            selector={
                "version": 1,
                "final": True,
                "partition_count": 2,
                "selected": {"1": {"attempt_id": 1, "split": selected_split}},
            },
        )

    handle.update_fte_exchange_selector(
        "query-fte-selector-final-coverage",
        "query-fte-selector-final-coverage:node:8",
        "3",
        selector={
            "partition_count": 2,
            "selected": {"1": {"attempt_id": 1, "split": selected_split}},
        },
    )
    selector_stats = handle.fte_registry_stats()["queries"]["query-fte-selector-final-coverage"]["fragment_executions"][
        "query-fte-selector-final-coverage:node:8"
    ]["exchange_selectors"]["3"]
    assert selector_stats["version"] == 0


def test_fte_exchange_selector_final_version_replay_is_idempotent(monkeypatch):
    monkeypatch.setattr(
        RayWorkerActorHandle,
        "_fte_task_handle_cls",
        staticmethod(lambda: _FakeFteTaskHandle),
    )
    actor = _FakeActor()
    handle = RayWorkerActorHandle(actor, memory_capacity_bytes=1 << 60)
    first_task = _FakeTask(
        name="exchange-selector-final-version-0",
        context={"query_id": "query-fte-selector-final-version", "node_id": "8"},
        inputs={
            "3": {
                "kind": "exchange_source_task",
                "data": {
                    "partition_indices": [0],
                    "source_partition_count": 2,
                    "source_task_count": 1,
                    "source_handles": [{"partition_id": 0, "attempt_id": 1, "path": "selected-0"}],
                },
            }
        },
        plan={"plan": "downstream-template"},
    )
    split0 = {
        "source_node_id": "3",
        "sequence_id": 1,
        "kind": "exchange_source_task",
        "data": {
            "partition_indices": [0],
            "source_partition_count": 2,
            "source_task_count": 1,
            "source_handles": [{"partition_id": 0, "attempt_id": 1, "path": "selected-0"}],
        },
        "source_partition_id": 0,
    }
    split1 = {
        "source_node_id": "3",
        "sequence_id": 2,
        "kind": "exchange_source_task",
        "data": {
            "partition_indices": [1],
            "source_partition_count": 2,
            "source_task_count": 1,
            "source_handles": [{"partition_id": 1, "attempt_id": 1, "path": "selected-1"}],
        },
        "source_partition_id": 1,
    }
    selector = {
        "version": 1,
        "final": True,
        "partition_count": 2,
        "selected": {
            "0": {"attempt_id": 1, "split": split0},
            "1": {"attempt_id": 1, "split": split1},
        },
    }

    first_handles = handle.submit_tasks([first_task])
    actor.fte_calls.clear()
    handles = handle.update_fte_exchange_selector(
        "query-fte-selector-final-version",
        "query-fte-selector-final-version:node:8",
        "3",
        selector=selector,
    )

    assert len(first_handles) == 1
    assert handles == []
    assert [call[0] for call in actor.fte_calls] == [
        "wait_split_queue",
        "add_splits",
        "no_more_splits",
    ]
    selector_stats = handle.fte_registry_stats()["queries"]["query-fte-selector-final-version"]["fragment_executions"][
        "query-fte-selector-final-version:node:8"
    ]["exchange_selectors"]["3"]
    assert selector_stats["version"] == 1
    assert selector_stats["final"] is True
    assert selector_stats["partition_count"] == 2
    assert selector_stats["selected_partitions"] == [0, 1]

    actor.fte_calls.clear()
    replay_handles = handle.update_fte_exchange_selector(
        "query-fte-selector-final-version",
        "query-fte-selector-final-version:node:8",
        "3",
        selector=selector,
    )

    assert replay_handles == []
    assert actor.fte_calls == []


def test_fte_exchange_selector_event_final_replay_is_idempotent(monkeypatch):
    monkeypatch.setattr(
        RayWorkerActorHandle,
        "_fte_task_handle_cls",
        staticmethod(lambda: _FakeFteTaskHandle),
    )
    actor = _FakeActor()
    handle = RayWorkerActorHandle(actor, memory_capacity_bytes=1 << 60)
    first_task = _FakeTask(
        name="exchange-selector-final-replay-0",
        context={"query_id": "query-fte-selector-final-replay", "node_id": "8"},
        inputs={
            "3": {
                "kind": "exchange_source_task",
                "data": {
                    "partition_indices": [0],
                    "source_partition_count": 1,
                    "source_task_count": 1,
                },
            }
        },
        plan={"plan": "downstream-template"},
    )

    first_handles = handle.submit_tasks([first_task])
    actor.fte_calls.clear()
    final_handles = handle.update_fte_exchange_selector(
        "query-fte-selector-final-replay",
        "query-fte-selector-final-replay:node:8",
        "3",
        selector=_exchange_selector_payload(
            final=True,
            partition_count=1,
            selected={"0": None},
        ),
    )
    assert len(first_handles) == 1
    assert final_handles == []
    assert [call[0] for call in actor.fte_calls] == ["no_more_splits"]

    actor.fte_calls.clear()
    replay_handles = handle.update_fte_exchange_selector(
        "query-fte-selector-final-replay",
        "query-fte-selector-final-replay:node:8",
        "3",
        selector=_exchange_selector_payload(
            final=True,
            partition_count=1,
            selected={"0": None},
        ),
    )

    assert replay_handles == []
    assert actor.fte_calls == []

    with pytest.raises(ValueError, match="cannot update final exchange selector"):
        handle.update_fte_exchange_selector(
            "query-fte-selector-final-replay",
            "query-fte-selector-final-replay:node:8",
            "3",
            selector=_exchange_selector_payload(
                [
                    {
                        "source_node_id": "3",
                        "sequence_id": 9,
                        "kind": "exchange_source_task",
                        "data": {
                            "partition_indices": [0],
                            "source_partition_count": 1,
                            "source_task_count": 1,
                        },
                        "source_partition_id": 0,
                    }
                ],
                partition_count=1,
            ),
        )
    assert actor.fte_calls == []


def test_fte_exchange_source_descriptor_replay_is_idempotent(monkeypatch):
    monkeypatch.setattr(
        RayWorkerActorHandle,
        "_fte_task_handle_cls",
        staticmethod(lambda: _FakeFteTaskHandle),
    )
    actor = _FakeActor()
    handle = RayWorkerActorHandle(actor, memory_capacity_bytes=1 << 60)
    first_task = _FakeTask(
        name="exchange-task-idempotent-0",
        context={"query_id": "query-fte-exchange-idempotent", "node_id": "8"},
        inputs={
            "3": {
                "kind": "exchange_source_task",
                "data": {
                    "partition_indices": [0],
                    "source_partition_count": 2,
                    "source_task_count": 1,
                },
            }
        },
        plan={"plan": "exchange-template"},
    )
    second_task = _FakeTask(
        name="exchange-task-idempotent-1",
        context={"query_id": "query-fte-exchange-idempotent", "node_id": "8"},
        inputs={
            "3": {
                "kind": "exchange_source_task",
                "data": {
                    "partition_indices": [1],
                    "source_partition_count": 2,
                    "source_task_count": 1,
                },
            }
        },
        plan={"plan": "exchange-template"},
    )
    final_replay = _FakeTask(
        name="exchange-task-idempotent-final-replay",
        context={"query_id": "query-fte-exchange-idempotent", "node_id": "8"},
        inputs={
            "3": {
                "kind": "exchange_source_task",
                "data": {
                    "partition_indices": [0, 1],
                    "source_partition_count": 2,
                    "source_task_count": 1,
                },
            }
        },
        plan={"plan": "exchange-template"},
    )

    first_handles = handle.submit_tasks([first_task])
    actor.fte_calls.clear()
    partial_replay_handles = handle.submit_tasks([first_task])
    assert partial_replay_handles == []
    assert actor.fte_calls == []

    final_handles = handle.submit_tasks([second_task])
    assert final_handles == []
    assert [call[0] for call in actor.fte_calls] == [
        "wait_split_queue",
        "add_splits",
    ]

    actor.fte_calls.clear()
    final_replay_handles = handle.submit_tasks([final_replay])

    assert len(first_handles) == 1
    assert final_replay_handles == []
    assert actor.fte_calls == []


def test_fte_exchange_source_same_partition_accepts_new_handle_batch(monkeypatch):
    monkeypatch.setattr(
        RayWorkerActorHandle,
        "_fte_task_handle_cls",
        staticmethod(lambda: _FakeFteTaskHandle),
    )
    actor = _FakeActor()
    handle = RayWorkerActorHandle(actor, memory_capacity_bytes=1 << 60)
    first_task = _FakeTask(
        name="exchange-task-same-partition-0",
        context={"query_id": "query-fte-exchange-same-partition", "node_id": "8"},
        inputs={
            "3": {
                "kind": "exchange_source_task",
                "data": {
                    "partition_indices": [0],
                    "source_partition_count": 1,
                    "source_task_count": 1,
                    "source_handles": [{"partition_id": 0, "path": "sink-a"}],
                },
            }
        },
        plan={"plan": "exchange-template"},
    )
    second_task = _FakeTask(
        name="exchange-task-same-partition-1",
        context={"query_id": "query-fte-exchange-same-partition", "node_id": "8"},
        inputs={
            "3": {
                "kind": "exchange_source_task",
                "data": {
                    "partition_indices": [0],
                    "source_partition_count": 1,
                    "source_task_count": 1,
                    "source_handles": [{"partition_id": 0, "path": "sink-b"}],
                },
            }
        },
        plan={"plan": "exchange-template"},
    )

    first_handles = handle.submit_tasks([first_task])
    actor.fte_calls.clear()
    second_handles = handle.submit_tasks([second_task])

    assert len(first_handles) == 1
    assert second_handles == []
    assert [call[0] for call in actor.fte_calls] == [
        "wait_split_queue",
        "add_splits",
    ]
    assert [split["source_partition_id"] for split in actor.fte_calls[1][3]] == [0]


def test_fte_input_stream_exhausted_sends_no_more(monkeypatch):
    monkeypatch.setattr(
        RayWorkerActorHandle,
        "_fte_task_handle_cls",
        staticmethod(lambda: _FakeFteTaskHandle),
    )
    actor = _FakeActor()
    handle = RayWorkerActorHandle(actor, memory_capacity_bytes=1 << 60)
    task = _FakeTask(
        name="scan-task",
        context={"query_id": "query-fte", "node_id": "7"},
        inputs={"7": {"kind": "scan_task", "data": b"a"}},
    )

    handle.submit_tasks([task])
    handles = handle.task_input_stream_exhausted(["7"])

    assert actor.fte_calls[-1][0] == "no_more_splits"
    assert actor.fte_calls[-1][2] == "7"
    assert handles == []
    stats = handle.fte_registry_stats()["event_schedulers"]["query-fte"]
    assert stats["event_counts"] == {
        "SplitEventsSubmitted": 1,
        "WorkerReservationCompleted": 1,
        "SourceInputExhausted": 1,
    }


def test_fte_attempt_create_starts_status_watcher(monkeypatch):
    monkeypatch.setattr(
        RayWorkerActorHandle,
        "_start_fte_attempt_status_watcher",
        _ORIGINAL_START_FTE_ATTEMPT_STATUS_WATCHER,
    )
    monkeypatch.setattr(
        RayWorkerActorHandle,
        "_fte_task_handle_cls",
        staticmethod(lambda: _FakeFteTaskHandle),
    )
    actor = _FakeActor()
    handle = RayWorkerActorHandle(actor, memory_capacity_bytes=1 << 60)
    task = _FakeTask(
        name="scan-task",
        context={"query_id": "query-fte-watcher", "node_id": "7"},
        inputs={"7": {"kind": "scan_task", "data": b"a"}},
    )

    running = handle.submit_tasks([task])[0]

    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        stats = handle.fte_registry_stats()["event_schedulers"]["query-fte-watcher"]
        if stats["event_counts"].get("TaskStatusChanged") == 1:
            break
        time.sleep(0.01)
    else:
        raise AssertionError("status watcher did not publish terminal task status")

    assert ("wait_status", running.task_id.to_dict(), None, 1.0) in actor.fte_calls
    query_status = handle.fte_query_status("query-fte-watcher")
    assert query_status["finished"] is True
    assert handle.fte_registry_stats()["status_watcher_count"] == 0


def test_fte_status_handler_keeps_watcher_until_terminal_status(monkeypatch):
    from vane.runners.ray import fragment_worker_events as worker_events_mod

    query_id = "query-fte-live-status"
    fragment_id = f"{query_id}:node:7"
    attempt_id = FteTaskAttemptId.coerce(
        {
            "query_id": query_id,
            "fragment_execution_id": 0,
            "partition_id": 0,
            "attempt_id": 0,
        }
    )

    class _Watcher:
        def __init__(self):
            self.stop_count = 0

        def stop(self):
            self.stop_count += 1

    class _FragmentExecution:
        def __init__(self):
            self.statuses = []

        def handle_task_status(self, status):
            self.statuses.append(dict(status))
            return None

    actor = _FakeActor()
    handle = RayWorkerActorHandle(actor, memory_capacity_bytes=1 << 60)
    watcher = _Watcher()
    fragment_execution = _FragmentExecution()
    monkeypatch.setattr(
        worker_events_mod,
        "fragment_execution_key_for_fte_attempt",
        lambda _attempt_id: (query_id, fragment_id),
    )
    worker_handle_mod._FTE_STATUS_WATCHERS[str(attempt_id)] = watcher
    worker_handle_mod._FTE_FRAGMENT_EXECUTIONS[(query_id, fragment_id)] = fragment_execution
    try:
        handle._handles_for_task_status_changed_event(
            TaskStatusChanged.from_status(
                query_id,
                attempt_id,
                {"state": "RUNNING", "task_stats": {"processed_input_rows": 5}},
            )
        )
        assert watcher.stop_count == 0
        assert fragment_execution.statuses[-1]["task_stats"]["processed_input_rows"] == 5

        handle._handles_for_task_status_changed_event(
            TaskStatusChanged.from_status(
                query_id,
                attempt_id,
                {"state": "FINISHED", "task_stats": {"processed_input_rows": 10}},
            )
        )
        assert watcher.stop_count == 1
        assert [status["state"] for status in fragment_execution.statuses] == [
            "RUNNING",
            "FINISHED",
        ]
    finally:
        worker_handle_mod._FTE_STATUS_WATCHERS.pop(str(attempt_id), None)
        worker_handle_mod._FTE_FRAGMENT_EXECUTIONS.pop((query_id, fragment_id), None)


def test_fte_status_watcher_registry_is_not_dropped_while_thread_is_alive():
    from vane.runners.ray.fte_scheduler import (
        FteAttemptStatusWatcher,
        FteSchedulerRegistry,
    )

    entered = threading.Event()

    class _SlowWorker:
        worker_id = "worker-slow-watcher-drop"

        def fte_wait_task_status(self, task_id, min_version, timeout_s):
            entered.set()
            time.sleep(0.25)
            return {
                "state": "RUNNING",
                "task_id": task_id,
                "version": 1,
            }

    query_id = "query-fte-slow-watcher-drop"
    scheduler = FteSchedulerRegistry().get_or_create(query_id)
    attempt_id = FteTaskAttemptId.coerce(
        {
            "query_id": query_id,
            "fragment_execution_id": 0,
            "partition_id": 0,
            "attempt_id": 0,
        }
    )
    watcher = FteAttemptStatusWatcher(
        scheduler=scheduler,
        attempt_id=attempt_id,
        worker=_SlowWorker(),
        wait_timeout_s=1.0,
        poll_interval_s=0.001,
    )
    worker_handle_mod._FTE_STATUS_WATCHERS[str(attempt_id)] = watcher

    assert watcher.start() is True
    assert entered.wait(1.0)
    worker_handle_mod._stop_fte_status_watchers(query_id)

    assert watcher.is_alive() is False
    assert str(attempt_id) not in worker_handle_mod._FTE_STATUS_WATCHERS


def test_status_watcher_drop_uses_exact_query_identity():
    class _Watcher:
        def __init__(self):
            self.stopped = False

        def stop(self):
            self.stopped = True

        def join(self, _timeout):
            pass

        def is_alive(self):
            return False

        def shutdown_timeout_s(self):
            return 1.0

    parent_key = "q.0.0.0"
    child_key = "q.child.0.0.0"
    parent = _Watcher()
    child = _Watcher()
    worker_handle_mod._FTE_STATUS_WATCHERS[parent_key] = parent
    worker_handle_mod._FTE_STATUS_WATCHERS[child_key] = child
    try:
        worker_handle_mod._stop_fte_status_watchers("q")

        assert parent.stopped is True
        assert child.stopped is False
        assert parent_key not in worker_handle_mod._FTE_STATUS_WATCHERS
        assert worker_handle_mod._FTE_STATUS_WATCHERS[child_key] is child
    finally:
        worker_handle_mod._FTE_STATUS_WATCHERS.pop(parent_key, None)
        worker_handle_mod._FTE_STATUS_WATCHERS.pop(child_key, None)


def test_worker_pressure_drop_uses_exact_query_identity():
    from vane.runners.ray.fragment_registry import _FteWorkerPressure
    from vane.runners.ray.fragment_worker_pressure import partition_reservation_key

    pressure = _FteWorkerPressure()
    parent_attempt = "q.0.0.0"
    child_attempt = "q.child.0.0.0"
    parent_reservation = partition_reservation_key("q", "q:node:1", 0)
    child_reservation = partition_reservation_key("q|child", "q|child:node:1", 0)
    pressure.running_attempts.update({parent_attempt, child_attempt})
    pressure.split_counts_by_attempt.update({parent_attempt: 1, child_attempt: 2})
    pressure.reserved_partitions.update({parent_reservation, child_reservation})
    pressure.memory_bytes_by_reservation.update({parent_reservation: 10, child_reservation: 20})

    pressure.drop_query("q")

    assert pressure.running_attempts == {child_attempt}
    assert pressure.split_counts_by_attempt == {child_attempt: 2}
    assert pressure.reserved_partitions == {child_reservation}
    assert pressure.memory_bytes_by_reservation == {child_reservation: 20}


def test_fte_status_watcher_rejects_mismatched_status_identity():
    from vane.runners.ray.fte_scheduler import (
        FteAttemptStatusWatcher,
        FteEventHandlers,
        FteSchedulerRegistry,
    )

    class _MismatchedWorker:
        worker_id = "worker-mismatched-watcher-status"

        def fte_wait_task_status(self, _task_id, _min_version, _timeout_s):
            return {
                "state": "FINISHED",
                "task_id_string": "query-fte-watcher-identity.0.99.0",
                "version": 1,
            }

    query_id = "query-fte-watcher-identity"
    scheduler = FteSchedulerRegistry().get_or_create(query_id)
    worker_failures = []
    terminal_statuses = []
    scheduler.set_handlers(
        FteEventHandlers(
            on_worker_failed=lambda event: worker_failures.append(event),
            on_task_status_changed=lambda event: terminal_statuses.append(event),
        )
    )
    attempt_id = FteTaskAttemptId.coerce(
        {
            "query_id": query_id,
            "fragment_execution_id": 0,
            "partition_id": 0,
            "attempt_id": 0,
        }
    )
    watcher = FteAttemptStatusWatcher(
        scheduler=scheduler,
        attempt_id=attempt_id,
        worker=_MismatchedWorker(),
        wait_timeout_s=1.0,
        poll_interval_s=0.001,
    )

    watcher.start()
    watcher.join(1.0)

    assert watcher.is_alive() is False
    assert len(worker_failures) == 1
    assert "status identity mismatch" in str(worker_failures[0].error)
    assert terminal_statuses == []


def test_fte_registry_close_waits_for_terminal_handler_and_suppresses_retry(monkeypatch):
    from vane.runners.ray import fragment_worker_events as worker_events_mod
    from vane.runners.ray.fte_scheduler import FteAttemptStatusWatcher

    query_id = "query-fte-close-terminal-race"
    fragment_id = f"{query_id}:node:7"
    attempt_id = FteTaskAttemptId.coerce(
        {
            "query_id": query_id,
            "fragment_execution_id": 0,
            "partition_id": 0,
            "attempt_id": 0,
        }
    )
    handler_entered = threading.Event()
    release_handler = threading.Event()
    close_done = threading.Event()
    retry_attempts = []
    outbox_executions = []

    class _TerminalWorker:
        worker_id = "worker-close-terminal-race"

        def fte_wait_task_status(self, task_id, _min_version, _timeout_s):
            return {
                "state": "FAILED",
                "task_id": task_id,
                "failure": {"message": "retryable"},
                "version": 1,
            }

    class _BlockedFragmentExecution:
        def handle_task_status(self, _status):
            handler_entered.set()
            assert release_handler.wait(2.0)
            return object()

    actor = _FakeActor()
    handle = RayWorkerActorHandle(actor, memory_capacity_bytes=1 << 60)
    monkeypatch.setattr(
        worker_events_mod,
        "fragment_execution_key_for_fte_attempt",
        lambda _attempt_id: (query_id, fragment_id),
    )
    monkeypatch.setattr(
        handle,
        "_execute_fte_fragment_execution_outbox",
        lambda execution: outbox_executions.append(execution),
    )
    monkeypatch.setattr(
        handle,
        "_handles_for_fte_scheduled_attempts",
        lambda *_args: retry_attempts.append(_args) or [],
    )
    worker_handle_mod._FTE_FRAGMENT_EXECUTIONS[(query_id, fragment_id)] = _BlockedFragmentExecution()
    scheduler = worker_handle_mod._FTE_SCHEDULERS.get_or_create(query_id)
    handle._bind_fte_scheduler_handlers(scheduler)
    watcher = FteAttemptStatusWatcher(
        scheduler=scheduler,
        attempt_id=attempt_id,
        worker=_TerminalWorker(),
        wait_timeout_s=1.0,
        poll_interval_s=0.001,
    )

    def unregister(exited_watcher):
        with worker_handle_mod._FTE_REGISTRY_LOCK:
            if worker_handle_mod._FTE_STATUS_WATCHERS.get(str(attempt_id)) is exited_watcher:
                worker_handle_mod._FTE_STATUS_WATCHERS.pop(str(attempt_id), None)

    watcher.on_exit = unregister
    worker_handle_mod._FTE_STATUS_WATCHERS[str(attempt_id)] = watcher
    watcher.start()
    assert handler_entered.wait(1.0)

    def close_registry():
        worker_handle_mod.close_fte_registry_for_query(query_id)
        worker_handle_mod.quiesce_fte_registry_for_query(query_id)
        close_done.set()

    close_thread = threading.Thread(target=close_registry)
    close_thread.start()
    time.sleep(0.05)
    assert close_done.is_set() is False

    release_handler.set()
    close_thread.join(2.0)

    assert close_done.is_set() is True
    assert watcher.is_alive() is False
    assert retry_attempts == []
    assert outbox_executions == []


def test_fte_registry_close_fences_inflight_remote_mutation_ownership():
    query_id = "query-fte-close-operation-fence"
    close_done = threading.Event()

    assert worker_handle_mod.begin_fte_registry_operation(query_id) is True

    def close_registry():
        worker_handle_mod.close_fte_registry_for_query(query_id)
        worker_handle_mod.quiesce_fte_registry_for_query(query_id)
        close_done.set()

    close_thread = threading.Thread(target=close_registry)
    close_thread.start()
    time.sleep(0.05)

    assert close_done.is_set() is False
    assert worker_handle_mod.begin_fte_registry_operation(query_id) is False

    worker_handle_mod.end_fte_registry_operation(query_id)
    close_thread.join(2.0)

    assert close_done.is_set() is True
    worker_handle_mod.open_fte_registry_for_query(query_id)
    assert worker_handle_mod.begin_fte_registry_operation(query_id) is True
    worker_handle_mod.end_fte_registry_operation(query_id)


def test_fragment_registration_ownership_lives_until_remote_actor_completion():
    class _DeferredFuture:
        def __init__(self):
            self._callbacks = []
            self._done = False

        def add_done_callback(self, callback):
            if self._done:
                callback(self)
            else:
                self._callbacks.append(callback)

        def complete(self):
            self._done = True
            callbacks = list(self._callbacks)
            self._callbacks.clear()
            for callback in callbacks:
                callback(self)

        def result(self):
            return {"registered": 1}

    class _DeferredObjectRef:
        def __init__(self, future):
            self._future = future

        def future(self):
            return self._future

    class _DeferredRemoteMethod:
        def __init__(self, object_ref):
            self.object_ref = object_ref
            self.calls = []

        def remote(self, payload):
            self.calls.append(payload)
            return self.object_ref

    query_id = "query-fragment-registration-fence"
    deferred_future = _DeferredFuture()
    actor = _FakeActor()
    actor.register_fragments = _DeferredRemoteMethod(_DeferredObjectRef(deferred_future))
    handle = RayWorkerActorHandle(actor, memory_capacity_bytes=1 << 60)
    task = _FakeTask(
        name="scan-task",
        context={"query_id": query_id, "node_id": "7"},
        inputs={"7": {"kind": "scan_task", "data": b"a"}},
    )

    handle.submit_tasks([task])

    assert worker_handle_mod._FTE_ACTIVE_OPERATIONS_BY_QUERY[query_id] == 1
    close_done = threading.Event()

    def close_registry():
        worker_handle_mod.close_fte_registry_for_query(query_id)
        worker_handle_mod.quiesce_fte_registry_for_query(query_id)
        close_done.set()

    close_thread = threading.Thread(target=close_registry)
    close_thread.start()
    time.sleep(0.05)
    assert close_done.is_set() is False

    deferred_future.complete()
    close_thread.join(2.0)

    assert close_done.is_set() is True
    assert query_id not in worker_handle_mod._FTE_ACTIVE_OPERATIONS_BY_QUERY


def test_failed_fragment_registration_does_not_suppress_retry():
    class _FailedFuture:
        def add_done_callback(self, callback):
            callback(self)

        def result(self):
            raise RuntimeError("planned fragment registration failure")

    class _FailedObjectRef:
        def future(self):
            return _FailedFuture()

    class _FailedRemoteMethod:
        def __init__(self):
            self.calls = []

        def remote(self, payload):
            self.calls.append(payload)
            return _FailedObjectRef()

    query_id = "query-fragment-registration-retry"
    fragment_id = f"{query_id}:node:7"
    actor = _FakeActor()
    actor.register_fragments = _FailedRemoteMethod()
    handle = RayWorkerActorHandle(actor, memory_capacity_bytes=1 << 60)

    handle.ensure_fragment_registered(query_id, fragment_id, {"plan": "fake"})
    handle.ensure_fragment_registered(query_id, fragment_id, {"plan": "fake"})

    assert len(actor.register_fragments.calls) == 2
    assert fragment_id not in handle._registered_fragment_ids
    assert query_id not in worker_handle_mod._FTE_ACTIVE_OPERATIONS_BY_QUERY


def test_pending_fragment_registration_is_reused_as_direct_dependency():
    query_id = "query-fragment-registration-pending-direct"
    fragment_id = f"{query_id}:node:7"
    pending_ref = object()
    actor = _FakeActor()
    handle = RayWorkerActorHandle(actor, memory_capacity_bytes=1 << 60)
    with handle._fragment_registration_lock:
        handle._registered_fragment_ids.add(fragment_id)
        handle._fragment_registration_refs[fragment_id] = pending_ref
        handle._fragment_query_ids[fragment_id] = query_id

    result = handle.ensure_fragment_registered(
        query_id,
        fragment_id,
        None,
    )

    assert result is pending_ref
    assert actor.register_payloads == []


def test_fragment_registration_cleanup_uses_exact_query_ownership():
    actor = _FakeActor()
    handle = RayWorkerActorHandle(actor, memory_capacity_bytes=1 << 60)
    parent_fragment = "q:node:1"
    child_fragment = "q:child:node:1"
    with handle._fragment_registration_lock:
        handle._registered_fragment_ids.update({parent_fragment, child_fragment})
        handle._fragment_query_ids.update({parent_fragment: "q", child_fragment: "q:child"})

    handle._drop_fragment_registration_state("q")

    assert handle._registered_fragment_ids == {child_fragment}
    assert handle._fragment_query_ids == {child_fragment: "q:child"}


def test_bulk_submit_reuses_pending_fragment_registration_dependency():
    query_id = "query-fragment-registration-pending-bulk"
    task = _FakeTask(
        name="scan-task",
        context={"query_id": query_id, "node_id": "7"},
        inputs={"7": {"kind": "scan_task", "data": b"a"}},
    )
    _, fragment_id = fragment_id_for_task(task.context(), task.name())
    pending_ref = object()
    actor = _FakeActor()
    handle = RayWorkerActorHandle(actor, memory_capacity_bytes=1 << 60)
    with handle._fragment_registration_lock:
        handle._registered_fragment_ids.add(fragment_id)
        handle._fragment_registration_refs[fragment_id] = pending_ref
        handle._fragment_query_ids[fragment_id] = query_id

    handle.submit_tasks([task])

    requests = _create_requests(actor)
    assert len(requests) == 1
    assert requests[0]["fragment_registration_result"] is pending_ref
    assert requests[0].get("fragment_plan") is None
    assert actor.register_payloads == []
    assert task.plan_calls == 0


def test_direct_fte_drop_waits_for_registry_fence_before_remote_drop():
    query_id = "query-direct-drop-fence"
    actor = _FakeActor()
    handle = RayWorkerActorHandle(actor, memory_capacity_bytes=1 << 60)
    drop_done = threading.Event()
    outcomes = []

    assert worker_handle_mod.begin_fte_registry_operation(query_id) is True

    def drop_query():
        try:
            outcomes.append(handle.fte_drop_query(query_id))
        except BaseException as exc:
            outcomes.append(exc)
        finally:
            drop_done.set()

    drop_thread = threading.Thread(target=drop_query)
    drop_thread.start()
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        if query_id in worker_handle_mod._FTE_CLOSING_QUERIES:
            break
        time.sleep(0.005)
    assert query_id in worker_handle_mod._FTE_CLOSING_QUERIES
    time.sleep(0.05)

    assert drop_done.is_set() is False
    assert not any(call[0] == "drop_query" for call in actor.fte_calls)

    worker_handle_mod.end_fte_registry_operation(query_id)
    drop_thread.join(2.0)

    assert drop_done.is_set() is True
    assert not isinstance(outcomes[0], BaseException)
    assert ("drop_query", query_id) in actor.fte_calls


def test_fte_attempt_handle_registered_before_status_watcher_start(monkeypatch):
    monkeypatch.setattr(
        RayWorkerActorHandle,
        "_start_fte_attempt_status_watcher",
        _ORIGINAL_START_FTE_ATTEMPT_STATUS_WATCHER,
    )
    monkeypatch.setattr(
        RayWorkerActorHandle,
        "_fte_task_handle_cls",
        staticmethod(lambda: _FakeFteTaskHandle),
    )
    observed_registered_ids = []

    def assert_registered_before_start(_self, query_id, attempt_id, _worker_handle):
        stored = worker_handle_mod._FTE_RESULT_HANDLES_BY_QUERY.get(str(query_id), [])
        stored_ids = [str(task_handle.task_id) for task_handle in stored]
        observed_registered_ids.append(stored_ids)
        assert str(attempt_id) in stored_ids

    monkeypatch.setattr(
        RayWorkerActorHandle,
        "_start_fte_attempt_status_watcher",
        assert_registered_before_start,
    )
    actor = _FakeActor()
    handle = RayWorkerActorHandle(actor, memory_capacity_bytes=1 << 60)
    task = _FakeTask(
        name="scan-task",
        context={"query_id": "query-fte-watcher-order", "node_id": "7"},
        inputs={"7": {"kind": "scan_task", "data": b"a"}},
    )

    running = handle.submit_tasks([task])[0]

    assert observed_registered_ids == [[str(running.task_id)]]


def test_fte_task_status_event_marks_partition_finished(monkeypatch):
    monkeypatch.setattr(
        RayWorkerActorHandle,
        "_fte_task_handle_cls",
        staticmethod(lambda: _FakeFteTaskHandle),
    )
    actor = _FakeActor()
    handle = RayWorkerActorHandle(actor, memory_capacity_bytes=1 << 60)
    task = _FakeTask(
        name="scan-task",
        context={"query_id": "query-fte-status", "node_id": "7"},
        inputs={"7": {"kind": "scan_task", "data": b"a"}},
    )

    running = handle.submit_tasks([task])[0]
    handles = handle.handle_fte_task_status(
        {
            "state": "FINISHED",
            "task_id": running.task_id.to_dict(),
            "version": 1,
        }
    )

    assert handles == []
    stage = worker_handle_mod._FTE_FRAGMENT_EXECUTIONS[("query-fte-status", "query-fte-status:node:7")]
    assert stage.partitions[0].finished is True
    stats = handle.fte_registry_stats()["event_schedulers"]["query-fte-status"]
    assert stats["event_counts"] == {
        "SplitEventsSubmitted": 1,
        "WorkerReservationCompleted": 1,
        "TaskStatusChanged": 1,
    }


def test_fte_task_status_event_retries_failed_attempt(monkeypatch):
    monkeypatch.setenv("VANE_FTE_RETRY_INITIAL_DELAY_S", "0")
    monkeypatch.setattr(
        RayWorkerActorHandle,
        "_fte_task_handle_cls",
        staticmethod(lambda: _FakeFteTaskHandle),
    )
    actor = _FakeActor()
    handle = RayWorkerActorHandle(actor, memory_capacity_bytes=1 << 60, worker_id="worker-0")
    task = _FakeTask(
        name="scan-task",
        context={"query_id": "query-fte-status-retry", "node_id": "7"},
        inputs={"7": {"kind": "scan_task", "data": b"a"}},
    )

    first = handle.submit_tasks([task])[0]
    retry_handles = handle.handle_fte_task_status(
        {
            "state": "FAILED",
            "task_id": first.task_id.to_dict(),
            "failure": {"message": "retry me"},
            "version": 1,
        }
    )

    assert len(retry_handles) == 1
    assert str(retry_handles[0].task_id) == "query-fte-status-retry.0.0.1"
    create_requests = [call[1] for call in actor.fte_calls if call[0] == "create"]
    assert [request["task_id"]["attempt_id"] for request in create_requests] == [0, 1]
    assert create_requests[1]["initial_splits"]["7"][0]["data"] == b"a"
    stats = handle.fte_registry_stats()["event_schedulers"]["query-fte-status-retry"]
    assert stats["event_counts"] == {
        "SplitEventsSubmitted": 1,
        "WorkerReservationCompleted": 2,
        "TaskStatusChanged": 1,
    }


def test_fte_task_status_event_oom_is_terminal_for_registered_heap(monkeypatch):
    monkeypatch.setattr(
        RayWorkerActorHandle,
        "_fte_task_handle_cls",
        staticmethod(lambda: _FakeFteTaskHandle),
    )
    actor = _FakeActor()
    handle = RayWorkerActorHandle(actor, memory_capacity_bytes=100, worker_id="worker-0")
    task = _FakeTask(
        name="scan-task",
        context={"query_id": "query-fte-oom-status", "node_id": "7"},
        inputs={"7": {"kind": "scan_task", "data": b"a"}},
    )

    first = handle.submit_tasks([task])[0]
    retry_handles = handle.handle_fte_task_status(
        {
            "state": "FAILED",
            "task_id": first.task_id.to_dict(),
            "failure": {
                "error_code": "EXCEEDED_LOCAL_MEMORY_LIMIT",
                "peak_memory_bytes": 1536,
            },
            "version": 1,
        }
    )

    assert retry_handles == []
    create_requests = [call[1] for call in actor.fte_calls if call[0] == "create"]
    assert [request["memory_requirement_bytes"] for request in create_requests] == [10]
    stats = handle.fte_registry_stats()["event_schedulers"]["query-fte-oom-status"]
    assert stats["event_counts"] == {
        "SplitEventsSubmitted": 1,
        "WorkerReservationCompleted": 1,
        "TaskStatusChanged": 1,
    }


def test_fte_wait_query_finishes_from_status_events(monkeypatch):
    monkeypatch.setattr(
        RayWorkerActorHandle,
        "_fte_task_handle_cls",
        staticmethod(lambda: _FakeFteTaskHandle),
    )
    monkeypatch.setattr(
        fragment_submission_mod,
        "_split_exchange_source_task_by_partition",
        lambda value: ([(int(value[-1:]), value)], 2, 2, False),
    )
    actor = _FakeActor()
    handle = RayWorkerActorHandle(actor, memory_capacity_bytes=1 << 60, worker_id="worker-0")
    tasks = [
        _FakeTask(
            name="exchange-task-0",
            context={"query_id": "query-fte-wait", "node_id": "8"},
            inputs={"3": {"kind": "exchange_source_task", "data": b"p0"}},
            plan={"plan": "exchange-template"},
        ),
        _FakeTask(
            name="exchange-task-1",
            context={"query_id": "query-fte-wait", "node_id": "8"},
            inputs={"3": {"kind": "exchange_source_task", "data": b"p1"}},
            plan={"plan": "exchange-template"},
        ),
    ]

    running = handle.submit_tasks(tasks)
    assert handle.fte_query_status("query-fte-wait")["finished"] is False
    for task_handle in running:
        handle.handle_fte_task_status(
            {
                "state": "FINISHED",
                "task_id": task_handle.task_id.to_dict(),
                "version": 1,
            }
        )

    status = handle.wait_fte_query("query-fte-wait", timeout_s=0)

    assert status["finished"] is True
    assert status["partition_count"] == 2
    assert status["finished_count"] == 2
    assert status["running_count"] == 0
    result_handles = handle.pop_fte_result_handles("query-fte-wait")
    assert [str(task_handle.task_id) for task_handle in result_handles] == [
        "query-fte-wait.0.0.0",
        "query-fte-wait.0.1.0",
    ]
    assert handle.pop_fte_result_handles("query-fte-wait") == []


def test_fte_wait_query_raises_on_failed_partition(monkeypatch):
    monkeypatch.setattr(
        RayWorkerActorHandle,
        "_fte_task_handle_cls",
        staticmethod(lambda: _FakeFteTaskHandle),
    )
    actor = _FakeActor()
    handle = RayWorkerActorHandle(actor, memory_capacity_bytes=1 << 60, worker_id="worker-0")
    task = _FakeTask(
        name="scan-task",
        context={"query_id": "query-fte-wait-failed", "node_id": "7"},
        inputs={"7": {"kind": "scan_task", "data": b"a"}},
    )

    running = handle.submit_tasks([task])[0]
    handle.handle_fte_task_status(
        {
            "state": "FAILED",
            "task_id": running.task_id.to_dict(),
            "failure": {"message": "not retryable", "retryable": False},
            "version": 1,
        }
    )

    with pytest.raises(RuntimeError, match="query-fte-wait-failed"):
        handle.wait_fte_query("query-fte-wait-failed", timeout_s=0)


def test_fte_input_stream_exhausted_seals_running_speculative_as_standard(monkeypatch):
    monkeypatch.setattr(
        RayWorkerActorHandle,
        "_fte_task_handle_cls",
        staticmethod(lambda: _FakeFteTaskHandle),
    )
    actor = _FakeActor()
    handle = RayWorkerActorHandle(actor, memory_capacity_bytes=15, worker_id="worker-0")
    task = _FakeTask(
        name="scan-task",
        context={
            "query_id": "query-fte-speculative-seal",
            "node_id": "7",
            "task_execution_class": "SPECULATIVE",
        },
        inputs={"7": {"kind": "scan_task", "data": b"a"}},
    )

    running = handle.submit_tasks([task])[0]

    assert isinstance(running, _FakeFteTaskHandle)
    assert handle.fte_pressure_stats()["speculative_memory_bytes"] == 10
    handles = handle.task_input_stream_exhausted(["7"])

    assert handles == []
    create_requests = [call[1] for call in actor.fte_calls if call[0] == "create"]
    assert create_requests[0]["execution_class"] == "SPECULATIVE"
    assert actor.fte_calls[-1][0] == "no_more_splits"
    stats = handle.fte_pressure_stats()
    assert stats["standard_memory_bytes"] == 10
    assert stats["speculative_memory_bytes"] == 0


def test_fte_dynamic_exchange_defaults_to_speculative_until_eof(monkeypatch):
    monkeypatch.setattr(
        RayWorkerActorHandle,
        "_fte_task_handle_cls",
        staticmethod(lambda: _FakeFteTaskHandle),
    )
    monkeypatch.setattr(
        fragment_submission_mod,
        "_split_exchange_source_task_by_partition",
        lambda value: ([(0, value)], 2, 1, False),
    )
    actor = _FakeActor()
    handle = RayWorkerActorHandle(actor, memory_capacity_bytes=15, worker_id="worker-0")
    task = _FakeTask(
        name="exchange-task",
        context={"query_id": "query-fte-exchange-speculative", "node_id": "8"},
        inputs={"3": {"kind": "exchange_source_task", "data": b"p0"}},
        plan={"plan": "exchange-template"},
    )

    running = handle.submit_tasks([task])[0]

    assert isinstance(running, _FakeFteTaskHandle)
    create_requests = [call[1] for call in actor.fte_calls if call[0] == "create"]
    assert create_requests[0]["execution_class"] == "SPECULATIVE"
    assert handle.fte_pressure_stats()["speculative_memory_bytes"] == 10
    handles = handle.task_input_stream_exhausted(["3"])

    assert handles == []
    stats = handle.fte_pressure_stats()
    assert stats["standard_memory_bytes"] == 10
    assert stats["speculative_memory_bytes"] == 0


def test_fte_input_stream_exhausted_control_failure_replays_sealed_descriptor(monkeypatch):
    monkeypatch.setattr(
        RayWorkerActorHandle,
        "_fte_task_handle_cls",
        staticmethod(lambda: _FakeFteTaskHandle),
    )

    class _DeadOnNoMoreActor(_FakeActor):
        def _fte_no_more_splits(self, task_id, source_node_id):
            self.fte_calls.append(("no_more_splits", task_id, source_node_id))
            raise RuntimeError("actor died during no_more_splits")

    actor0 = _DeadOnNoMoreActor()
    actor1 = _FakeActor()
    handle0 = RayWorkerActorHandle(actor0, memory_capacity_bytes=1 << 60, worker_id="worker-0")
    handle1 = RayWorkerActorHandle(actor1, memory_capacity_bytes=1 << 60, worker_id="worker-1")
    task = _FakeTask(
        name="scan-task",
        context={"query_id": "query-fte-no-more-lost", "node_id": "7"},
        inputs={"7": {"kind": "scan_task", "data": b"a"}},
        plan={"plan": "scan-template"},
    )

    first = handle0.submit_tasks([task])
    retries = handle0.task_input_stream_exhausted(["7"])

    assert len(first) == 1
    assert first[0].worker_handle is handle0
    assert len(retries) == 1
    assert retries[0].worker_handle is handle1
    assert (
        worker_handle_mod._FTE_PARTITION_OWNERS[("query-fte-no-more-lost", "query-fte-no-more-lost:node:7", 0)]
        is handle1
    )
    retry_creates = [call for call in actor1.fte_calls if call[0] == "create"]
    assert len(retry_creates) == 1
    retry_request = retry_creates[0][1]
    assert retry_request["task_id"]["attempt_id"] == 1
    assert retry_request["no_more_splits"] == ["7"]
    assert [split["data"] for split in retry_request["initial_splits"]["7"]] == [b"a"]
    stats = handle0.fte_registry_stats()["event_schedulers"]["query-fte-no-more-lost"]
    assert stats["event_counts"] == {
        "SplitEventsSubmitted": 1,
        "WorkerReservationCompleted": 2,
        "SourceInputExhausted": 1,
        "WorkerFailed": 1,
    }


def test_fte_empty_input_creates_task_instead_of_empty_sentinel(monkeypatch):
    monkeypatch.setattr(
        RayWorkerActorHandle,
        "_fte_task_handle_cls",
        staticmethod(lambda: _FakeFteTaskHandle),
    )
    actor = _FakeActor()
    handle = RayWorkerActorHandle(actor, memory_capacity_bytes=1 << 60)
    task = _FakeTask(
        name="aggregate-task",
        context={"query_id": "query-fte-empty", "node_id": "9"},
    )

    handles = handle.submit_tasks([task])

    assert isinstance(handles[0], _FakeFteTaskHandle)
    assert [call[0] for call in actor.fte_calls] == ["create"]
    assert actor.fte_calls[0][1]["initial_splits"] == {}


def test_submit_tasks_rejects_legacy_in_memory_task_inputs():
    actor = _FakeActor()
    handle = RayWorkerActorHandle(actor, memory_capacity_bytes=1 << 60)
    fragment_a = object()
    fragment_b = object()
    task = _FakeTask(
        name="in-memory-task-1",
        context={"query_id": "query-pset", "node_id": "9"},
        inputs={
            "11": {"kind": "in_memory_data", "fragments": [fragment_a, fragment_b]},
        },
        plan={"plan": "in-memory"},
    )

    with pytest.raises(ValueError, match="Unsupported task input kind"):
        handle.submit_tasks([task])

    assert actor.register_payloads == []
    assert actor.fragment_calls == []


def test_submit_tasks_extracts_exchange_source_task_inputs(monkeypatch):
    monkeypatch.setattr(
        fragment_submission_mod,
        "_split_exchange_source_task_by_partition",
        lambda value: ([(0, value)], 1, 1, False),
    )
    actor = _FakeActor()
    handle = RayWorkerActorHandle(actor, memory_capacity_bytes=1 << 60)
    task0 = _FakeTask(
        name="exchange-source-task-0",
        context={"query_id": "query-exchange", "node_id": "9"},
        inputs={
            "9": {"kind": "exchange_source_task", "data": b"binding-a"},
        },
        plan={"plan": "exchange-template"},
    )
    task1 = _FakeTask(
        name="exchange-source-task-1",
        context={"query_id": "query-exchange", "node_id": "9"},
        inputs={
            "9": {"kind": "exchange_source_task", "data": b"binding-b"},
        },
        plan={"plan": "exchange-template"},
    )
    expected_fragment_id = fragment_id_for_task(task0.context(), task0.name())[1]

    handles = handle.submit_tasks([task0, task1])

    assert len(handles) == 1
    request = _create_requests(actor)[0]
    assert request["fragment_id"] == expected_fragment_id
    assert "exchange_source_task:9" not in request["context"]
    assert "exchange_source_task_nodes" not in request["context"]
    assert request["dynamic_exchange_source_node_ids"] == ["9"]
    assert [split["data"] for split in request["initial_splits"]["9"]] == [
        b"binding-a",
        b"binding-b",
    ]
    assert actor.register_payloads == [
        [
            {
                "fragment_id": expected_fragment_id,
                "plan": {"plan": "exchange-template"},
                "query_id": "query-exchange",
            }
        ]
    ]
    assert actor.fragment_calls == []


def test_ray_worker_actor_class_cloudpickle_roundtrip():
    actor_cls = worker_mod.RayWorkerActor.__ray_metadata__.modified_class

    payload = ray.cloudpickle.dumps(actor_cls)
    restored = ray.cloudpickle.loads(payload)

    assert restored.__name__ == actor_cls.__name__


def test_ray_worker_fte_admission_log_uses_worker_id(monkeypatch, capsys):
    actor_cls = worker_mod.RayWorkerActor.__ray_metadata__.modified_class
    actor = object.__new__(actor_cls)
    actor._fte_task_manager = None
    actor._fte_admission_config = FteWorkerAdmissionConfig(
        max_running_tasks=4,
        mode="lease",
        memory_budget_bytes=16,
        task_memory_bytes=None,
    )
    actor._set_duckdb_memory_baseline = lambda _memory_bytes: None

    async def execute_fte_request(request):
        return {"ok": request["task_id"]}

    actor._execute_fte_request = execute_fte_request
    monkeypatch.setenv("VANE_FTE_ADMISSION_DEBUG", "1")
    monkeypatch.setenv("VANE_WORKER_ID", "ray-worker-log")

    async def run():
        task_id = {
            "query_id": "ray-log",
            "fragment_execution_id": 0,
            "partition_id": 0,
            "attempt_id": 0,
        }
        manager = actor._get_fte_task_manager()
        status = await manager.create_task(
            {
                "task_id": task_id,
                "fragment_id": "ray-log:node:scan",
                "memory_requirement_bytes": 4,
                "query_task_lease": {
                    "lease_id": "lease-ray-log",
                    "query_id": "ray-log",
                    "execution_query_id": "ray-log",
                    "stage_id": "stage:ray-log:node:scan:fte",
                    "task_id": "ray-log.0.0",
                    "attempt_id": "ray-log.0.0.0",
                    "resources": {
                        "cpu": 1.0,
                        "gpu": 0.0,
                        "heap_bytes": 4,
                        "object_store_bytes": 0,
                    },
                },
            }
        )
        assert status["state"] == FteTaskState.RUNNING.value
        for _ in range(50):
            status = await manager.get_task_status(task_id)
            if status["state"] == FteTaskState.FINISHED.value:
                break
            await asyncio.sleep(0.01)
        await asyncio.sleep(0)
        assert status["state"] == FteTaskState.FINISHED.value

    asyncio.run(run())
    captured = capsys.readouterr().err

    assert "[vane-fte-admission" in captured
    assert "worker_id=ray-worker-log" in captured
    assert "event=manager_init" in captured
    assert "event=create_task" in captured
    assert "event=start_task" in captured
    assert "event=task_done" in captured
    assert "task_id=ray-log.0.0.0" in captured
    assert "max_running=4" in captured


def test_drop_query_fragments_clears_local_registry():
    actor = _FakeActor()
    handle = RayWorkerActorHandle(actor, memory_capacity_bytes=1 << 60)
    handle._registered_fragment_ids = {"query-1:node:1", "query-2:node:2"}
    handle._fragment_query_ids = {
        "query-1:node:1": "query-1",
        "query-2:node:2": "query-2",
    }
    worker_handle_mod._FTE_RESULT_HANDLES_BY_QUERY["query-1"] = [object()]
    worker_handle_mod._FTE_RESULT_HANDLES_BY_QUERY["query-2"] = [object()]

    removed = handle.drop_query_fragments("query-1")

    assert removed == 1
    assert actor.drop_calls == ["query-1"]
    assert handle._registered_fragment_ids == {"query-2:node:2"}
    assert "query-1" not in worker_handle_mod._FTE_RESULT_HANDLES_BY_QUERY
    assert "query-2" in worker_handle_mod._FTE_RESULT_HANDLES_BY_QUERY


def test_drop_query_fragments_remote_failure_retains_retryable_local_registry():
    class _DeadDropActor(_FakeActor):
        def __init__(self):
            super().__init__()
            self.fail_once = True

        def _drop_query_fragments(self, query_id):
            self.drop_calls.append(query_id)
            if self.fail_once:
                self.fail_once = False
                raise RuntimeError("fragment actor is dead")
            return 1

    actor = _DeadDropActor()
    handle = RayWorkerActorHandle(actor, memory_capacity_bytes=1 << 60)
    handle._registered_fragment_ids = {
        "query-dead:node:1",
        "query-keep:node:2",
    }
    handle._fragment_query_ids = {
        "query-dead:node:1": "query-dead",
        "query-keep:node:2": "query-keep",
    }
    worker_handle_mod._FTE_RESULT_HANDLES_BY_QUERY["query-dead"] = [object()]
    worker_handle_mod._FTE_RESULT_HANDLES_BY_QUERY["query-keep"] = [object()]

    with pytest.raises(RuntimeError, match="fragment actor is dead"):
        handle.drop_query_fragments("query-dead")

    assert actor.drop_calls == ["query-dead"]
    assert handle._registered_fragment_ids == {
        "query-dead:node:1",
        "query-keep:node:2",
    }
    assert "query-dead" in worker_handle_mod._FTE_RESULT_HANDLES_BY_QUERY

    assert handle.drop_query_fragments("query-dead") == 1

    assert actor.drop_calls == ["query-dead", "query-dead"]
    assert handle._registered_fragment_ids == {"query-keep:node:2"}
    assert "query-dead" not in worker_handle_mod._FTE_RESULT_HANDLES_BY_QUERY
    assert "query-keep" in worker_handle_mod._FTE_RESULT_HANDLES_BY_QUERY


def test_stats_fragments_reads_worker_actor_counters():
    actor = _FakeActor()
    handle = RayWorkerActorHandle(actor, memory_capacity_bytes=1 << 60)

    stats = handle.stats_fragments()

    assert stats == {"registered_total": 2, "existing_total": 1, "lookup_hits": 3}
    assert actor.fragment_stats_calls == 1


def test_register_fragments_awaits_plan_refs_without_ray_get(monkeypatch):
    actor_cls = worker_mod.RayWorkerActor.__ray_metadata__.modified_class
    actor = object.__new__(actor_cls)
    actor._plan_fragments = {}
    actor._query_fragments = {}
    actor._fragment_query_ids = {}
    actor._fragment_register_calls = 0
    actor._fragment_registered_total = 0
    actor._fragment_existing_total = 0

    class _AwaitablePlanRef:
        def __init__(self, value):
            self.value = value
            self.awaited = False

        def __await__(self):
            async def _resolve():
                self.awaited = True
                return self.value

            return _resolve().__await__()

    monkeypatch.setattr(worker_mod.ray, "ObjectRef", _AwaitablePlanRef)
    monkeypatch.setattr(
        worker_mod.ray,
        "get",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("ray.get must not be used")),
    )

    class _Plan:
        def has_root(self):
            return True

    resolved_plan = _Plan()
    plan_ref = _AwaitablePlanRef(resolved_plan)

    result = asyncio.run(
        actor_cls.register_fragments(
            actor,
            [{"fragment_id": "query-1:node:1", "plan": plan_ref, "query_id": "query-1"}],
        )
    )

    assert result == {
        "registered": 1,
        "existing": 0,
        "total": 1,
    }
    assert plan_ref.awaited
    assert actor._plan_fragments["query-1:node:1"] is resolved_plan
    assert actor._query_fragments == {"query-1": {"query-1:node:1"}}


def test_start_ray_workers_skips_blocking_warmup_inside_ray_worker(monkeypatch):
    get_calls = []
    option_calls = []
    remote_calls = []

    class _FakeInstalledMethod:
        def remote(self, *_args, **_kwargs):
            return "warmup-ref"

    class _FakeActorHandle:
        def __init__(self):
            self.install_env_overrides = _FakeInstalledMethod()

    class _FakeActorFactory:
        def options(self, **kwargs):
            option_calls.append(kwargs)
            return self

        def remote(self, **kwargs):
            remote_calls.append(kwargs)
            return _FakeActorHandle()

    monkeypatch.setattr(worker_handle_mod, "_is_ray_worker_context", lambda: True)
    monkeypatch.setattr(worker_handle_mod, "_collect_vane_env_overrides", dict)
    monkeypatch.setattr(worker_handle_mod, "RayWorkerActor", _FakeActorFactory())
    monkeypatch.setattr(
        worker_handle_mod.ray,
        "nodes",
        lambda: [
            {
                "NodeID": "node-a",
                "NodeManagerAddress": "10.0.0.1",
                "Resources": {"CPU": 4.0, "memory": 1024.0, "GPU": 0.0},
            }
        ],
    )
    monkeypatch.setattr(
        worker_handle_mod.ray,
        "get",
        lambda value, *args, **kwargs: get_calls.append((value, args, kwargs)) or value,
    )
    monkeypatch.setattr(
        worker_handle_mod.ray.util.scheduling_strategies,
        "NodeAffinitySchedulingStrategy",
        lambda **kwargs: ("node-affinity", kwargs),
    )

    runtimes = worker_handle_mod.start_ray_workers(existing_worker_ids=[])

    assert len(runtimes) == 1
    assert option_calls[0]["memory"] == 358
    assert len(remote_calls) == 1
    assert remote_calls[0]["env_overrides"]["VANE_WORKER_ID"] == "10.0.0.1"
    assert remote_calls[0]["env_overrides"]["VANE_WORKER_INDEX"] == "0"
    assert remote_calls[0]["duckdb_memory_bytes"] == 256
    assert remote_calls[0]["task_heap_capacity_bytes"] == 615
    assert get_calls == []


def test_execute_native_task_passes_exchange_and_sink_inputs():
    actor_cls = worker_mod.RayWorkerActor.__ray_metadata__.modified_class
    actor = object.__new__(actor_cls)
    calls = []

    class _FakeCursor:
        def close(self):
            return None

    class _FakeConn:
        def __init__(self):
            self.cursor_obj = _FakeCursor()

        def cursor(self):
            return self.cursor_obj

        def close(self):
            return None

    class _FakePlanRunner:
        def execute_native(
            self,
            cursor,
            plan,
            scan_task_arg,
            exchange_source_task_arg,
            copy_output_info,
            exchange_sink_instance,
            fte_scan_source_queues,
            fte_exchange_source_queues,
            dynamic_filter_domains,
            native_progress_callback,
            runtime_context,
        ):
            calls.append(
                (
                    cursor,
                    plan,
                    scan_task_arg,
                    exchange_source_task_arg,
                    copy_output_info,
                    exchange_sink_instance,
                    fte_scan_source_queues,
                    fte_exchange_source_queues,
                    dynamic_filter_domains,
                    native_progress_callback,
                    runtime_context,
                )
            )
            return "ok"

    shared_conn = _FakeConn()
    actor._get_shared_conn = lambda: shared_conn
    actor._get_plan_runner = lambda: _FakePlanRunner()

    dynamic_domains = {"df0": {"column": "id", "single_value": 7}}
    result = actor_cls._execute_native_task(
        actor,
        "fake-plan",
        {"1": b"scan"},
        copy_output_info={"base": "", "run_id": "run-native", "remote_base": "/tmp/out"},
        exchange_source_task_map={"9": b"exchange-binding"},
        exchange_sink_instance={"sink_handle": {"partition_id": 4}, "attempt_id": 2, "attempt_path": "/tmp/attempt"},
        dynamic_filter_domains=dynamic_domains,
        debug_context={"query_id": "q1", "fragment_id": "f1", "task_id": "q1.2.3.4"},
    )

    assert result == "ok"
    assert len(calls) == 1
    (
        _,
        plan,
        scan_task_arg,
        exchange_source_task_arg,
        copy_output_info,
        exchange_sink_instance,
        fte_scan_source_queues,
        fte_exchange_source_queues,
        dynamic_filter_domains,
        native_progress_callback,
        runtime_context,
    ) = calls[0]
    assert plan == "fake-plan"
    assert scan_task_arg == {"1": b"scan"}
    assert exchange_source_task_arg == {"9": b"exchange-binding"}
    assert copy_output_info == {"base": "", "run_id": "run-native", "remote_base": "/tmp/out"}
    assert exchange_sink_instance == {
        "sink_handle": {"partition_id": 4},
        "attempt_id": 2,
        "attempt_path": "/tmp/attempt",
    }
    assert fte_scan_source_queues is None
    assert fte_exchange_source_queues is None
    assert dynamic_filter_domains == dynamic_domains
    assert native_progress_callback is None
    assert runtime_context == {"query_id": "q1", "fragment_id": "f1", "task_id": "q1.2.3.4"}


def test_execute_native_task_uses_shared_database_for_fte():
    actor_cls = worker_mod.RayWorkerActor.__ray_metadata__.modified_class
    actor = object.__new__(actor_cls)
    calls = []
    closed = []

    class _FakeCursor:
        def close(self):
            closed.append("cursor")

    class _FakeConn:
        def __init__(self):
            self.cursor_obj = _FakeCursor()

        def cursor(self):
            return self.cursor_obj

        def close(self):
            closed.append("conn")

    shared_conn = _FakeConn()
    actor._get_shared_conn = lambda: shared_conn

    class _FakePlanRunner:
        def execute_native(
            self,
            cursor,
            _plan,
            _scan_task_arg,
            _exchange_source_task_arg,
            _copy_output_info,
            _exchange_sink_instance,
            fte_scan_source_queues,
            fte_exchange_source_queues,
            _dynamic_filter_domains,
            _native_progress_callback,
            _runtime_context,
        ):
            calls.append((cursor, fte_scan_source_queues, fte_exchange_source_queues))
            return "ok"

    shared_runner = _FakePlanRunner()
    actor._get_plan_runner = lambda: shared_runner

    scan_queues = {"1": object()}
    exchange_queues = {"2": object()}
    result = actor_cls._execute_native_task(
        actor,
        "fake-plan",
        None,
        fte_scan_source_queues=scan_queues,
        fte_exchange_source_queues=exchange_queues,
    )

    assert result == "ok"
    assert calls == [(shared_conn.cursor_obj, scan_queues, exchange_queues)]
    assert closed == ["cursor"]


def test_worker_del_closes_shared_connection():
    actor_cls = worker_mod.RayWorkerActor.__ray_metadata__.modified_class
    actor = object.__new__(actor_cls)
    calls = []

    class _FakeConn:
        def interrupt(self):
            calls.append("interrupt")

        def close(self):
            calls.append("close")

    actor._shared_conn = _FakeConn()

    actor_cls.__del__(actor)

    assert calls == ["interrupt", "close"]
    assert actor._shared_conn is None
