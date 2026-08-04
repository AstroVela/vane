# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from types import SimpleNamespace

import pytest

import duckdb

ray = pytest.importorskip("ray")

import duckdb.runners.fte.fte_execution as fte_execution_mod
import duckdb.runners.ray.fragment_worker_commands as worker_commands_mod
import duckdb.runners.ray.fragment_worker_failures as worker_failures_mod
import duckdb.runners.ray.fragment_worker_placement as worker_placement_mod
import duckdb.runners.ray.fragment_worker_selection as worker_selection_mod
import duckdb.runners.ray.fragment_worker_submission as fragment_submission_mod
import duckdb.runners.ray.fragment_worker_task_control as task_control_mod
import duckdb.runners.ray.fragment_worker_transitions as worker_transitions_mod
import duckdb.runners.ray.fte_fragment_scheduler as fte_fragment_scheduler_mod
import duckdb.runners.ray.worker as worker_mod
import duckdb.runners.ray.worker_handle as worker_handle_mod
from duckdb.runners.common import QueryDeadlineExceeded
from duckdb.runners.fte.fte_attempts import FragmentExecutionMutationResult
from duckdb.runners.fte.fte_config import FteWorkerAdmissionConfig
from duckdb.runners.ray.fragment_worker_context import fragment_id_for_task
from duckdb.runners.ray.fte import (
    AssignmentResult,
    FteFragmentExecution,
    FteTaskAttemptId,
    FteTaskExecution,
    FteTaskId,
    FteTaskState,
    FteWorkerReservationUnavailable,
    NodeRequirements,
    PartitionInfo,
)
from duckdb.runners.ray.fte_events import (
    FteAddSplitsCommand,
    FteCreateTaskCommand,
    FteNoMoreSplitsCommand,
    TaskStatusChanged,
    WorkerFailed,
    WorkerReservationCompleted,
)
from duckdb.runners.ray.query_execution_graph import (
    NodeResourceAllocation,
    QueryAllocation,
    QueryExecutionGraph,
    ResourceVector,
    StageResourceSpec,
)
from duckdb.runners.ray.query_graph_builder import fte_stage_id_for_fragment
from duckdb.runners.ray.query_resource_runtime import (
    clear_query_resource_managers,
    get_query_resource_manager,
    register_query_graph,
)
from duckdb.runners.ray.worker_handle import RayWorkerActorHandle as _ProductionRayWorkerActorHandle


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
        host=None,
        manager_instance_id=None,
    ):
        super().__init__(
            actor_handle,
            memory_capacity_bytes=memory_capacity_bytes,
            worker_id=str(worker_id or f"test-worker-{id(actor_handle)}"),
            node_id=str(node_id or _test_ray_node_id()),
            host=host,
            manager_instance_id=manager_instance_id,
        )

    def record_fte_task_result_ready(self, attempt_id):
        result = super().record_fte_task_result_ready(attempt_id)
        # These unit handles do not materialize FteWorkerTaskHandle results, so
        # model immediate result adoption before scheduling the next partition.
        self.record_fte_task_terminal(attempt_id)
        return result

    def record_fte_task_result_ready_without_drain(self, attempt_id):
        result = super().record_fte_task_result_ready_without_drain(attempt_id)
        # Match the immediate-adoption model without re-entering pending drain
        # while a fragment completion is being applied.
        self.record_fte_task_terminal(attempt_id, drain=False)
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
        self.shutdown_calls = []
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
        self.fte_prepare_drop_query = _FakeRemoteMethod(self._fte_drop_query)
        self.fte_cleanup_query = _FakeRemoteMethod(self._fte_cleanup_query)
        self.prepare_shutdown = _FakeRemoteMethod(self._prepare_shutdown)

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
        return {
            "has_space": True,
            "buffered_splits": 0,
            "status": {"state": FteTaskState.RUNNING.value, "task_id": task_id},
        }

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

    def _fte_cleanup_query(self, query_id):
        self.fte_calls.append(("cleanup_query", query_id))
        return {}

    def _prepare_shutdown(self):
        self.shutdown_calls.append("prepare")


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
    worker_handle_mod._FTE_STABLE_TASK_IDENTITY_KEYS_BY_RESOURCE_QUERY.clear()
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
    monkeypatch.setattr(worker_handle_mod.ray, "kill", lambda *_args, **_kwargs: None)
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


def test_fte_materialized_sink_identity_is_independent_of_fragment_registration_order():
    actor = _FakeActor()
    handle = RayWorkerActorHandle(actor, memory_capacity_bytes=1 << 60)
    first_query_id = "query-logical-fragment-order-a"
    second_query_id = "query-logical-fragment-order-b"
    node_ids = ("7", "42")
    for query_id in (first_query_id, second_query_id):
        _register_test_query_graph(
            query_id,
            [f"{query_id}:node:{node_id}" for node_id in node_ids],
        )

    def submit_fragments(query_id, ordered_node_ids):
        request_offset = len(_create_requests(actor))
        handles = handle.submit_tasks(
            [
                _FakeTask(
                    name=f"sample-input-{node_id}",
                    context={"query_id": query_id, "node_id": node_id},
                    inputs={node_id: {"kind": "scan_task", "data": node_id.encode()}},
                    exchange_sink_instance={
                        "sink_handle": {"task_partition_id": 0, "partition_id": 0},
                        "task_partition_id": 0,
                        "partition_id": 0,
                        "attempt_id": 0,
                        "output_location": f"{query_id}_coordinator__sink_0__attempt_0",
                        "fte_task_identity": True,
                    },
                )
                for node_id in ordered_node_ids
            ]
        )
        assert len(handles) == len(ordered_node_ids)
        requests = _create_requests(actor)[request_offset:]
        return {
            request["fragment_id"].rsplit(":node:", 1)[1]: request
            for request in requests
            if request["task_id"]["query_id"] == query_id
        }

    first_by_node = submit_fragments(first_query_id, ("42", "7"))
    second_by_node = submit_fragments(second_query_id, ("7", "42"))

    for node_id in node_ids:
        assert (
            first_by_node[node_id]["exchange_sink_instance"]["task_partition_id"]
            == second_by_node[node_id]["exchange_sink_instance"]["task_partition_id"]
        )
    assert (
        first_by_node["7"]["exchange_sink_instance"]["task_partition_id"]
        != first_by_node["42"]["exchange_sink_instance"]["task_partition_id"]
    )
    assert (
        first_by_node["7"]["task_id"]["fragment_execution_id"]
        != second_by_node["7"]["task_id"]["fragment_execution_id"]
    )


def test_fte_materialized_sink_identity_distinguishes_explicit_fragments_in_one_stage():
    actor = _FakeActor()
    handle = RayWorkerActorHandle(actor, memory_capacity_bytes=1 << 60)

    def submit_fragments(resource_query_id, ordered_task_indices):
        execution_query_id = f"{resource_query_id}_orderby_42_final"
        resource_stage_id = f"stage:{resource_query_id}:node:42:fte"
        _register_test_query_graph(resource_query_id, [f"{resource_query_id}:node:42"])
        request_offset = len(_create_requests(actor))
        handles = handle.submit_tasks(
            [
                _FakeTask(
                    name="OrderByFinal",
                    context={
                        "query_id": execution_query_id,
                        "resource_query_id": resource_query_id,
                        "resource_stage_id": resource_stage_id,
                        "fragment_id": f"{execution_query_id}:orderby:42:OrderByFinal:{task_idx}",
                        "stable_task_partition_id": str(task_idx),
                    },
                    exchange_sink_instance={
                        "sink_handle": {"task_partition_id": 0, "partition_id": 0},
                        "task_partition_id": 0,
                        "partition_id": 0,
                        "attempt_id": 0,
                        "output_location": f"{execution_query_id}_coordinator__sink_0__attempt_0",
                        "fte_task_identity": True,
                    },
                )
                for task_idx in ordered_task_indices
            ]
        )
        assert len(handles) == len(ordered_task_indices)
        requests = _create_requests(actor)[request_offset:]
        return {request["fragment_id"].rsplit(":", 1)[1]: request for request in requests}

    first_by_task = submit_fragments("query-explicit-fragment-a", (1, 0))
    second_by_task = submit_fragments("query-explicit-fragment-b", (0, 1))

    for task_idx in ("0", "1"):
        assert (
            first_by_task[task_idx]["exchange_sink_instance"]["task_partition_id"]
            == second_by_task[task_idx]["exchange_sink_instance"]["task_partition_id"]
        )
    assert (
        first_by_task["0"]["exchange_sink_instance"]["task_partition_id"]
        != first_by_task["1"]["exchange_sink_instance"]["task_partition_id"]
    )
    assert (
        first_by_task["0"]["task_id"]["fragment_execution_id"]
        != second_by_task["0"]["task_id"]["fragment_execution_id"]
    )


def test_fte_stable_task_identity_registry_rejects_hash_collisions():
    fragment_submission_mod._register_fte_stable_task_identity("query-collision", 17, "logical-task-a")
    fragment_submission_mod._register_fte_stable_task_identity("query-collision", 17, "logical-task-a")

    with pytest.raises(ValueError, match="stable Ray FTE task identity collision"):
        fragment_submission_mod._register_fte_stable_task_identity("query-collision", 17, "logical-task-b")


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
        "cleanup_query",
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


def test_fte_split_backpressure_remote_error_is_canceled_by_query_close():
    query_id = "query-split-wait-close"
    wait_started = threading.Event()
    add_called = threading.Event()

    class _BackpressuredWorker:
        def fte_wait_split_queue_has_space(self, *_args):
            raise AssertionError("query-owned split wait must be interruptible")

        def fte_wait_split_queue_has_space_interruptible(
            self,
            _task_id,
            _source_node_id,
            _max_buffered_splits,
            _timeout_s,
            stop_event,
        ):
            wait_started.set()
            deadline = time.monotonic() + 1.0
            while not stop_event.is_set():
                if time.monotonic() >= deadline:
                    raise TimeoutError("query close did not cancel split wait")
                time.sleep(0.01)
            raise RuntimeError("remote split wait raced query close")

        def enqueue_fte_add_splits(self, *_args):
            add_called.set()
            raise AssertionError("canceled split submission must not reach the worker")

    handle = RayWorkerActorHandle(
        _FakeActor(),
        memory_capacity_bytes=1 << 60,
        worker_id="worker-split-wait-close",
    )
    stage = FteFragmentExecution(
        query_id,
        0,
        fragment_id=f"{query_id}:node:7",
        task_memory_bytes=1,
    )
    attempt_id = FteTaskAttemptId.coerce(
        {
            "query_id": query_id,
            "fragment_execution_id": 0,
            "partition_id": 0,
            "attempt_id": 0,
        }
    )
    command = FteAddSplitsCommand(
        query_id=query_id,
        fragment_id=stage.fragment_id,
        worker_id="worker-split-wait-close",
        worker=_BackpressuredWorker(),
        attempt_id=attempt_id,
        source_node_id="7",
        splits=({"sequence_id": 1, "kind": "scan_task", "data": b"a"},),
    )

    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            execution = executor.submit(
                handle._execute_fte_fragment_execution_worker_commands,
                stage,
                [command],
            )
            assert wait_started.wait(timeout=1.0)
            worker_handle_mod.close_fte_registry_for_query(query_id)
            execution.result(timeout=1.0)

        assert add_called.is_set() is False
        assert query_id not in worker_handle_mod._FTE_ACTIVE_OPERATIONS_BY_QUERY
    finally:
        fte_fragment_scheduler_mod._drop_fte_registry_for_query(query_id)


def test_fte_worker_command_dispatch_preserves_healthy_tail_and_new_outbox_commands(monkeypatch):
    query_id = "query-command-owned-tail"
    coordinator = RayWorkerActorHandle(
        _FakeActor(),
        memory_capacity_bytes=1 << 60,
        worker_id="worker-command-coordinator",
    )
    failed_a = RayWorkerActorHandle(
        _FakeActor(),
        memory_capacity_bytes=1 << 60,
        worker_id="worker-command-failed-a",
    )
    healthy = RayWorkerActorHandle(
        _FakeActor(),
        memory_capacity_bytes=1 << 60,
        worker_id="worker-command-healthy",
    )
    failed_b = RayWorkerActorHandle(
        _FakeActor(),
        memory_capacity_bytes=1 << 60,
        worker_id="worker-command-failed-b",
    )
    stage = FteFragmentExecution(
        query_id,
        0,
        fragment_id=f"{query_id}:node:7",
        task_memory_bytes=1,
    )

    def add_command(worker, partition_id):
        return FteAddSplitsCommand(
            query_id=query_id,
            fragment_id=stage.fragment_id,
            worker_id=worker.worker_id,
            worker=worker,
            attempt_id=FteTaskAttemptId(FteTaskId(query_id, 0, partition_id), 0),
            source_node_id="7",
            splits=({"sequence_id": partition_id, "kind": "scan_task", "data": b"x"},),
        )

    failed_first = add_command(failed_a, 0)
    failed_same_worker_tail = add_command(failed_a, 1)
    healthy_first = add_command(healthy, 2)
    failed_other_worker = add_command(failed_b, 3)
    healthy_second = add_command(healthy, 4)
    appended_during_dispatch = add_command(healthy, 5)
    stage._worker_command_outbox.extend(
        [
            failed_first,
            failed_same_worker_tail,
            healthy_first,
            failed_other_worker,
            healthy_second,
        ]
    )

    fte_fragment_scheduler_mod.open_fte_registry_for_query(query_id)
    scheduler = worker_commands_mod._FTE_SCHEDULERS.get_or_create(query_id)
    executed = []
    appended = False

    def execute(command, **_kwargs):
        nonlocal appended
        executed.append((command.worker_id, command.attempt_id.partition_id))
        if command is failed_first or command is failed_other_worker:
            raise RuntimeError(f"planned control failure for {command.worker_id}")
        if command is healthy_first:
            assert failed_a._fte_healthy is False
            if not appended:
                stage._record_worker_command(appended_during_dispatch)
                appended = True
        if command is healthy_second:
            assert failed_b._fte_healthy is False

    monkeypatch.setattr(scheduler.worker_command_executor, "execute", execute)

    try:
        dispatch = coordinator._execute_fte_fragment_execution_outbox(stage)

        assert executed == [
            (failed_a.worker_id, 0),
            (healthy.worker_id, 2),
            (failed_b.worker_id, 3),
            (healthy.worker_id, 4),
        ]
        assert dispatch.failed_worker_ids == {failed_a.worker_id, failed_b.worker_id}
        assert len(dispatch.failures) == 2
        assert dispatch.query_closed is False
        assert [command.attempt_id.partition_id for command in stage._worker_command_outbox] == [5]
        assert failed_a._fte_healthy is False
        assert failed_b._fte_healthy is False
        assert healthy._fte_healthy is True
        assert scheduler.stats().event_counts.get("WorkerFailed", 0) == 2
    finally:
        fte_fragment_scheduler_mod._drop_fte_registry_for_query(query_id)


def test_fte_worker_command_dispatch_publishes_only_successful_healthy_creates(monkeypatch):
    query_id = "query-command-create-outcomes"
    coordinator = RayWorkerActorHandle(
        _FakeActor(),
        memory_capacity_bytes=1 << 60,
        worker_id="worker-create-coordinator",
    )
    create_failed = RayWorkerActorHandle(
        _FakeActor(),
        memory_capacity_bytes=1 << 60,
        worker_id="worker-create-failed",
    )
    failed_after_create = RayWorkerActorHandle(
        _FakeActor(),
        memory_capacity_bytes=1 << 60,
        worker_id="worker-failed-after-create",
    )
    healthy = RayWorkerActorHandle(
        _FakeActor(),
        memory_capacity_bytes=1 << 60,
        worker_id="worker-create-healthy",
    )
    stage = FteFragmentExecution(
        query_id,
        0,
        fragment_id=f"{query_id}:node:7",
        task_memory_bytes=1,
    )

    scheduled_attempts = []
    create_commands = []
    for partition_id, worker in enumerate((create_failed, failed_after_create, healthy)):
        partition = stage.add_partition(partition_id)
        scheduled = partition.start_attempt(
            worker_id=worker.worker_id,
            worker_incarnation_id=worker.worker_incarnation_id,
            remote_handle=worker,
        )
        scheduled_attempts.append(scheduled)
        create_commands.append(
            FteCreateTaskCommand(
                query_id=query_id,
                fragment_id=stage.fragment_id,
                worker_id=worker.worker_id,
                worker=worker,
                attempt_id=scheduled.attempt_id,
                partition_id=partition_id,
                request=scheduled.request,
                scheduled_attempt=scheduled,
            )
        )
    fail_after_create_command = FteAddSplitsCommand(
        query_id=query_id,
        fragment_id=stage.fragment_id,
        worker_id=failed_after_create.worker_id,
        worker=failed_after_create,
        attempt_id=scheduled_attempts[1].attempt_id,
        source_node_id="7",
        splits=({"sequence_id": 1, "kind": "scan_task", "data": b"x"},),
    )
    mutation_result = FragmentExecutionMutationResult.from_attempts(
        scheduled_attempts,
        [
            create_commands[0],
            create_commands[1],
            fail_after_create_command,
            create_commands[2],
        ],
    )

    fte_fragment_scheduler_mod.open_fte_registry_for_query(query_id)
    scheduler = worker_commands_mod._FTE_SCHEDULERS.get_or_create(query_id)
    executed = []

    def execute(command, **_kwargs):
        executed.append((command.command_type, command.worker_id))
        if command is create_commands[0] or command is fail_after_create_command:
            raise RuntimeError(f"planned control failure for {command.worker_id}")

    monkeypatch.setattr(scheduler.worker_command_executor, "execute", execute)
    monkeypatch.setattr(worker_commands_mod, "fte_partition_task_lease_payload", lambda *_args: {})

    try:
        dispatch = coordinator._execute_fte_fragment_execution_mutation_result(stage, mutation_result)

        assert executed == [
            ("FteCreateTaskCommand", create_failed.worker_id),
            ("FteCreateTaskCommand", failed_after_create.worker_id),
            ("FteAddSplitsCommand", failed_after_create.worker_id),
            ("FteCreateTaskCommand", healthy.worker_id),
        ]
        assert dispatch.failed_worker_ids == {create_failed.worker_id, failed_after_create.worker_id}
        assert dispatch.scheduled_attempts == (scheduled_attempts[2],)
        assert str(scheduled_attempts[1].attempt_id) not in {
            str(attempt.attempt_id) for attempt in dispatch.scheduled_attempts
        }
        assert healthy.fte_pressure_stats()["running_attempt_count"] == 1
        assert scheduler.stats().event_counts.get("WorkerFailed", 0) == 2
    finally:
        fte_fragment_scheduler_mod._drop_fte_registry_for_query(query_id)


def test_fte_worker_command_dispatch_claim_owns_all_create_publication(monkeypatch):
    query_id = "query-command-create-claim"
    workers = [
        RayWorkerActorHandle(
            _FakeActor(),
            memory_capacity_bytes=1 << 60,
            worker_id=f"worker-create-claim-{partition_id}",
        )
        for partition_id in range(2)
    ]
    stage = FteFragmentExecution(
        query_id,
        0,
        fragment_id=f"{query_id}:node:7",
        worker_selector=lambda partition: (
            workers[partition.task_id.partition_id].worker_id,
            workers[partition.task_id.partition_id],
        ),
        task_memory_bytes=1,
    )
    coordinator = RayWorkerActorHandle(
        _FakeActor(),
        memory_capacity_bytes=1 << 60,
        worker_id="worker-create-claim-coordinator",
    )

    fte_fragment_scheduler_mod.open_fte_registry_for_query(query_id)
    scheduler = worker_commands_mod._FTE_SCHEDULERS.get_or_create(query_id)
    monkeypatch.setattr(scheduler.worker_command_executor, "execute", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(worker_commands_mod, "fte_partition_task_lease_payload", lambda *_args: {})

    try:
        first = stage.start_attempt_with_worker(stage.add_partition(0))
        second = stage.start_attempt_with_worker(stage.add_partition(1))

        claimed = coordinator._execute_fte_fragment_execution_outbox(stage)
        already_claimed = coordinator._execute_fte_fragment_execution_outbox(stage)

        assert claimed.scheduled_attempts == (first, second)
        assert already_claimed.scheduled_attempts == ()
        assert [worker.fte_pressure_stats()["running_attempt_count"] for worker in workers] == [1, 1]
    finally:
        fte_fragment_scheduler_mod._drop_fte_registry_for_query(query_id)


def test_fte_worker_command_dispatch_query_close_owns_successful_create(monkeypatch):
    query_id = "query-command-close-after-create"
    worker = RayWorkerActorHandle(
        _FakeActor(),
        memory_capacity_bytes=1 << 60,
        worker_id="worker-command-close-after-create",
    )
    stage = FteFragmentExecution(
        query_id,
        0,
        fragment_id=f"{query_id}:node:7",
        worker=worker,
        task_memory_bytes=1,
    )
    coordinator = RayWorkerActorHandle(
        _FakeActor(),
        memory_capacity_bytes=1 << 60,
        worker_id="worker-command-close-coordinator",
    )

    fte_fragment_scheduler_mod.open_fte_registry_for_query(query_id)
    scheduled = stage.start_attempt_with_worker(stage.add_partition(0))
    scheduler = worker_commands_mod._FTE_SCHEDULERS.get_or_create(query_id)
    monkeypatch.setattr(
        scheduler.worker_command_executor,
        "execute",
        lambda *_args, **_kwargs: worker_handle_mod.close_fte_registry_for_query(query_id),
    )
    monkeypatch.setattr(worker_commands_mod, "fte_partition_task_lease_payload", lambda *_args: {})

    try:
        dispatch = coordinator._execute_fte_fragment_execution_outbox(stage)

        assert dispatch.query_closed is True
        assert dispatch.scheduled_attempts == ()
        assert worker.fte_pressure_stats()["running_attempt_count"] == 1
        assert str(scheduled.attempt_id) not in worker_handle_mod._FTE_STATUS_WATCHERS
    finally:
        fte_fragment_scheduler_mod._drop_fte_registry_for_query(query_id)


def test_fte_split_backpressure_preserves_query_deadline():
    query_id = "query-split-wait-deadline"

    class _DeadlineWorker:
        def fte_wait_split_queue_has_space(self, *_args):
            raise AssertionError("query-owned split wait must be interruptible")

        def fte_wait_split_queue_has_space_interruptible(self, *_args):
            raise QueryDeadlineExceeded("query deadline expired while waiting for split capacity")

    handle = RayWorkerActorHandle(
        _FakeActor(),
        memory_capacity_bytes=1 << 60,
        worker_id="worker-split-wait-deadline",
    )
    stage = FteFragmentExecution(
        query_id,
        0,
        fragment_id=f"{query_id}:node:7",
        task_memory_bytes=1,
    )
    attempt_id = FteTaskAttemptId.coerce(
        {
            "query_id": query_id,
            "fragment_execution_id": 0,
            "partition_id": 0,
            "attempt_id": 0,
        }
    )
    command = FteAddSplitsCommand(
        query_id=query_id,
        fragment_id=stage.fragment_id,
        worker_id="worker-split-wait-deadline",
        worker=_DeadlineWorker(),
        attempt_id=attempt_id,
        source_node_id="7",
        splits=({"sequence_id": 1, "kind": "scan_task", "data": b"a"},),
    )

    try:
        with pytest.raises(QueryDeadlineExceeded, match="query deadline expired"):
            handle._execute_fte_fragment_execution_worker_commands(stage, [command])
        assert query_id not in worker_handle_mod._FTE_ACTIVE_OPERATIONS_BY_QUERY
        assert handle._fte_healthy is True
    finally:
        fte_fragment_scheduler_mod._drop_fte_registry_for_query(query_id)


def test_fte_split_backpressure_terminal_status_uses_task_status_path(monkeypatch):
    query_id = "query-split-wait-terminal"
    attempt_id = FteTaskAttemptId.coerce(
        {
            "query_id": query_id,
            "fragment_execution_id": 0,
            "partition_id": 0,
            "attempt_id": 0,
        }
    )
    terminal_status = {
        "state": FteTaskState.FAILED.value,
        "task_id": attempt_id.to_dict(),
        "failure": {
            "error_code": "GENERIC_INTERNAL_ERROR",
            "type": "ValueError",
            "message": "planned task failure",
        },
    }

    class _TerminalWorker:
        def fte_wait_split_queue_has_space(self, *_args):
            raise AssertionError("query-owned split wait must be interruptible")

        def fte_wait_split_queue_has_space_interruptible(self, *_args):
            return {
                "has_space": False,
                "terminal": True,
                "status": terminal_status,
            }

        def enqueue_fte_add_splits(self, *_args, **_kwargs):
            raise AssertionError("terminal task must not accept more splits")

        def enqueue_fte_no_more_splits(self, *_args, **_kwargs):
            raise AssertionError("terminal task must skip trailing controls")

    worker = _TerminalWorker()
    handle = RayWorkerActorHandle(
        _FakeActor(),
        memory_capacity_bytes=1 << 60,
        worker_id="worker-split-wait-terminal",
    )
    stage = FteFragmentExecution(
        query_id,
        0,
        fragment_id=f"{query_id}:node:7",
        task_memory_bytes=1,
    )
    add_command = FteAddSplitsCommand(
        query_id=query_id,
        fragment_id=stage.fragment_id,
        worker_id=handle.worker_id,
        worker=worker,
        attempt_id=attempt_id,
        source_node_id="7",
        splits=({"sequence_id": 1, "kind": "scan_task", "data": b"a"},),
    )
    no_more_command = FteNoMoreSplitsCommand(
        query_id=query_id,
        fragment_id=stage.fragment_id,
        worker_id=handle.worker_id,
        worker=worker,
        attempt_id=attempt_id,
        source_node_id="7",
    )
    captured_events = []
    monkeypatch.setattr(
        handle,
        "_handles_for_task_status_changed_event",
        lambda event: captured_events.append(event) or [],
    )
    worker_handle_mod.open_fte_registry_for_query(query_id)

    try:
        handle._execute_fte_fragment_execution_worker_commands(
            stage,
            [add_command, no_more_command],
        )
        scheduler = worker_commands_mod._FTE_SCHEDULERS.get(query_id)
        assert scheduler is not None
        scheduler.drain()

        assert len(captured_events) == 1
        assert captured_events[0].status == terminal_status
        assert handle._fte_healthy is True
        stats = scheduler.stats()
        assert stats.event_counts.get("TaskStatusChanged", 0) == 1
        assert stats.event_counts.get("WorkerFailed", 0) == 0
    finally:
        fte_fragment_scheduler_mod._drop_fte_registry_for_query(query_id)


@pytest.mark.parametrize(
    ("terminal_state", "failure"),
    [
        pytest.param(FteTaskState.FINISHED, None, id="finished"),
        pytest.param(
            FteTaskState.FAILED,
            {
                "error_code": "GENERIC_INTERNAL_ERROR",
                "message": "planned task failure",
            },
            id="failed",
        ),
        pytest.param(
            FteTaskState.CANCELED,
            {
                "error_code": "TASK_CANCELED",
                "message": "planned task cancellation",
            },
            id="canceled",
        ),
        pytest.param(
            FteTaskState.ABORTED,
            {
                "error_code": "TASK_ABORTED",
                "message": "planned task abort",
            },
            id="aborted",
        ),
    ],
)
def test_fte_late_add_splits_terminal_status_uses_task_status_path(
    monkeypatch,
    terminal_state,
    failure,
):
    query_id = f"query-late-add-splits-{terminal_state.value.lower()}"
    other_query_id = f"{query_id}-other"
    attempt_id = FteTaskAttemptId.coerce(
        {
            "query_id": query_id,
            "fragment_execution_id": 0,
            "partition_id": 0,
            "attempt_id": 0,
        }
    )

    async def execute_fn(_request):
        return None

    execution = FteTaskExecution(
        {
            "task_id": attempt_id.to_dict(),
            "dynamic_scan_source_node_ids": ["7"],
        },
        execute_fn,
        default_task_memory_bytes=1,
    )
    execution._transition(terminal_state, failure=failure)
    terminal_status = execution.status_payload()

    class _TaskManager:
        @staticmethod
        async def add_splits(task_id, source_node_id, splits):
            assert FteTaskAttemptId.coerce(task_id) == attempt_id
            assert source_node_id == "7"
            assert splits == [{"sequence_id": 1, "kind": "scan_task", "data": b"a"}]
            execution.add_splits(source_node_id, splits)
            raise AssertionError("terminal add_splits must not return normally")

    class _WorkerEndpoint:
        @staticmethod
        def _get_fte_task_manager():
            return _TaskManager()

    actor_class = worker_mod.RayWorkerActor.__ray_metadata__.modified_class

    class _LateTerminalAddActor(_FakeActor):
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
            return {
                "has_space": True,
                "terminal": False,
                "buffered_splits": 0,
                "status": {
                    "state": FteTaskState.RUNNING.value,
                    "task_id": task_id,
                },
            }

        def _fte_add_splits(self, task_id, source_node_id, splits, dependency=None):
            self.fte_calls.append(("add_splits", task_id, source_node_id, splits))
            return asyncio.run(
                actor_class.fte_add_splits(
                    _WorkerEndpoint(),
                    task_id,
                    source_node_id,
                    splits,
                    dependency,
                )
            )

        def _fte_no_more_splits(self, *_args, **_kwargs):
            raise AssertionError("terminal task must skip trailing controls")

    actor = _LateTerminalAddActor()
    handle = RayWorkerActorHandle(
        actor,
        memory_capacity_bytes=1 << 60,
        worker_id=f"worker-late-add-splits-{terminal_state.value.lower()}",
    )
    stage = FteFragmentExecution(
        query_id,
        0,
        fragment_id=f"{query_id}:node:7",
        task_memory_bytes=1,
    )
    add_command = FteAddSplitsCommand(
        query_id=query_id,
        fragment_id=stage.fragment_id,
        worker_id=handle.worker_id,
        worker=handle,
        attempt_id=attempt_id,
        source_node_id="7",
        splits=({"sequence_id": 1, "kind": "scan_task", "data": b"a"},),
    )
    no_more_command = FteNoMoreSplitsCommand(
        query_id=query_id,
        fragment_id=stage.fragment_id,
        worker_id=handle.worker_id,
        worker=handle,
        attempt_id=attempt_id,
        source_node_id="7",
    )
    captured_events = []
    monkeypatch.setattr(
        handle,
        "_handles_for_task_status_changed_event",
        lambda event: captured_events.append(event) or [],
    )
    worker_handle_mod.open_fte_registry_for_query(query_id)
    worker_handle_mod.open_fte_registry_for_query(other_query_id)
    other_scheduler = worker_commands_mod._FTE_SCHEDULERS.get_or_create(other_query_id)
    handle._bind_fte_scheduler_handlers(other_scheduler)

    try:
        handle._execute_fte_fragment_execution_worker_commands(
            stage,
            [add_command, no_more_command],
        )
        scheduler = worker_commands_mod._FTE_SCHEDULERS.get(query_id)
        assert scheduler is not None
        scheduler.drain()

        assert len(captured_events) == 1
        assert captured_events[0].status == terminal_status
        assert [call[0] for call in actor.fte_calls] == [
            "wait_split_queue",
            "add_splits",
        ]
        assert handle._fte_healthy is True
        with worker_handle_mod._FTE_REGISTRY_LOCK:
            assert worker_handle_mod._FTE_WORKER_HANDLES.get(handle.worker_id) is handle
        assert scheduler.stats().event_counts.get("WorkerFailed", 0) == 0
        assert other_scheduler.stats().event_counts.get("WorkerFailed", 0) == 0
        assert handle.fte_drop_query(query_id) == {
            "tasks_removed": 1,
            "tasks_canceled": 0,
            "fragments_removed": 2,
        }
        assert handle._has_fte_control_state_for_query(query_id) is False
        assert handle._has_fte_teardown_state_for_query(query_id) is False
        assert handle._fte_healthy is True
        with worker_handle_mod._FTE_REGISTRY_LOCK:
            assert worker_handle_mod._FTE_WORKER_HANDLES.get(handle.worker_id) is handle
        assert other_scheduler.stats().event_counts.get("WorkerFailed", 0) == 0
    finally:
        fte_fragment_scheduler_mod._drop_fte_registry_for_query(query_id)
        fte_fragment_scheduler_mod._drop_fte_registry_for_query(other_query_id)


def test_fte_late_add_splits_unknown_attempt_remains_strict_error():
    task_id = {
        "query_id": "query-late-add-splits-unknown",
        "fragment_execution_id": 0,
        "partition_id": 0,
        "attempt_id": 0,
    }

    class _TaskManager:
        @staticmethod
        async def add_splits(_task_id, _source_node_id, _splits):
            raise KeyError("unknown FTE task attempt")

    class _WorkerEndpoint:
        @staticmethod
        def _get_fte_task_manager():
            return _TaskManager()

    actor_class = worker_mod.RayWorkerActor.__ray_metadata__.modified_class

    with pytest.raises(KeyError, match="unknown FTE task attempt"):
        asyncio.run(
            actor_class.fte_add_splits(
                _WorkerEndpoint(),
                task_id,
                "7",
                [{"sequence_id": 1, "kind": "scan_task", "data": b"a"}],
            )
        )


def test_interruptible_fte_control_ref_preserves_completed_query_deadline():
    completed = Future()
    completed.set_exception(QueryDeadlineExceeded("remote query deadline"))
    ref = SimpleNamespace(future=lambda: completed)

    with pytest.raises(QueryDeadlineExceeded, match="remote query deadline"):
        task_control_mod.FteWorkerTaskControlMixin._get_fte_control_ref(
            "fte_wait_split_queue_has_space",
            ref,
            timeout_s=0.1,
            cancel_event=threading.Event(),
        )


def test_interruptible_fte_control_ref_reloads_result_completed_at_timeout_boundary():
    class _RacingFuture:
        def __init__(self):
            self.result_calls = []

        def result(self, timeout=None):
            self.result_calls.append(timeout)
            if len(self.result_calls) == 1:
                raise FutureTimeoutError
            return "completed"

        def done(self):
            return True

    future = _RacingFuture()
    ref = SimpleNamespace(future=lambda: future)

    assert (
        task_control_mod.FteWorkerTaskControlMixin._get_fte_control_ref(
            "fte_wait_split_queue_has_space",
            ref,
            timeout_s=0.1,
            cancel_event=threading.Event(),
        )
        == "completed"
    )
    assert future.result_calls == [0.05, None]


def test_fte_control_ref_preserves_query_deadline_when_wait_expires(monkeypatch):
    pending = Future()
    ref = SimpleNamespace(future=lambda: pending)
    monkeypatch.setenv("VANE_QUERY_DEADLINE_EPOCH_S", str(time.time() + 0.05))

    try:
        with pytest.raises(QueryDeadlineExceeded, match="query deadline expired"):
            task_control_mod.FteWorkerTaskControlMixin._get_fte_control_ref(
                "fte_add_splits",
                ref,
                timeout_s=30.0,
            )
    finally:
        pending.cancel()


def test_fte_cancel_barrier_outlives_configured_control_timeouts(monkeypatch):
    query_id = "query-cancel-barrier-timeout"
    task_id = {
        "query_id": query_id,
        "fragment_execution_id": 0,
        "partition_id": 0,
        "attempt_id": 1,
    }
    pending = Future()

    class _DeferredCancel:
        def options(self, **_kwargs):
            return self

        def remote(self, *_args):
            return SimpleNamespace(future=lambda: pending)

    actor = _FakeActor()
    actor.fte_cancel_task = _DeferredCancel()
    handle = RayWorkerActorHandle(actor, memory_capacity_bytes=1 << 60)
    monkeypatch.setenv("VANE_FTE_CONTROL_RPC_TIMEOUT_S", "0.05")
    monkeypatch.setenv("VANE_RAY_OBJECT_GET_TIMEOUT_S", "0.05")
    monkeypatch.setenv("VANE_QUERY_DEADLINE_EPOCH_S", str(time.time() - 1.0))
    cancel_results = []
    cancel_errors = []
    cancel_started = threading.Event()

    def cancel_task():
        cancel_started.set()
        try:
            cancel_results.append(handle.fte_cancel_task(task_id))
        except BaseException as exc:  # pragma: no cover - asserted below
            cancel_errors.append(exc)

    cancel_thread = threading.Thread(target=cancel_task)
    cancel_thread.start()
    assert cancel_started.wait(timeout=1.0)
    time.sleep(0.15)
    outlived_control_timeout = cancel_thread.is_alive()
    pending.set_result(
        {
            "state": FteTaskState.CANCELED.value,
            "task_id": task_id,
            "_fte_control_operation": "fte_cancel_task",
            "_fte_control_applied": True,
        }
    )
    cancel_thread.join(timeout=2.0)

    assert outlived_control_timeout
    assert not cancel_thread.is_alive()
    assert cancel_errors == []
    assert cancel_results[0]["state"] == FteTaskState.CANCELED.value


def test_ordered_add_ref_is_canceled_and_unowned_after_query_close(monkeypatch):
    query_id = "query-cancel-ordered-add"
    task_id = {
        "query_id": query_id,
        "fragment_execution_id": 0,
        "partition_id": 0,
        "attempt_id": 0,
    }
    add_started = threading.Event()
    pending = Future()

    class _DeferredRef:
        def future(self):
            return pending

    ref = _DeferredRef()

    class _DeferredAdd:
        def remote(self, *_args):
            add_started.set()
            return ref

    actor = _FakeActor()
    actor.fte_add_splits = _DeferredAdd()
    handle = RayWorkerActorHandle(actor, memory_capacity_bytes=1 << 60)
    canceled_refs = []
    barrier_snapshot_started = threading.Event()
    cancellation_applied = threading.Event()
    resolve_object_refs_blocking = task_control_mod.resolve_object_refs_blocking

    def cancel_ref(object_ref, *, force=False):
        assert force is False
        canceled_refs.append(object_ref)
        pending.cancel()
        cancellation_applied.set()

    def resolve_after_cancellation(*args, **kwargs):
        barrier_snapshot_started.set()
        assert cancellation_applied.wait(timeout=1.0)
        return resolve_object_refs_blocking(*args, **kwargs)

    monkeypatch.setattr(ray, "cancel", cancel_ref)
    monkeypatch.setattr(task_control_mod, "resolve_object_refs_blocking", resolve_after_cancellation)
    query_closing = worker_commands_mod._FteQueryClosingEvent(query_id)

    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            submitted = executor.submit(
                handle.enqueue_fte_add_splits,
                task_id,
                "7",
                [{"sequence_id": 1}],
                cancel_event=query_closing,
            )
            assert add_started.wait(timeout=1.0)
            assert worker_handle_mod._FTE_ACTIVE_OPERATIONS_BY_QUERY[query_id] == 1
            flushed = executor.submit(handle.close_and_flush_fte_controls, query_id)
            assert barrier_snapshot_started.wait(timeout=1.0)
            with pytest.raises(InterruptedError, match="interrupted by cancellation"):
                submitted.result(timeout=1.0)
            assert flushed.result(timeout=1.0) == []

        assert canceled_refs == [ref]
        assert str(FteTaskAttemptId.coerce(task_id)) not in handle._fte_control_tails_by_task
        assert query_id not in worker_handle_mod._FTE_ACTIVE_OPERATIONS_BY_QUERY
    finally:
        if not pending.done():
            pending.cancel()
        fte_fragment_scheduler_mod._drop_fte_registry_for_query(query_id)


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
    assert handle._has_fte_teardown_state_for_query("q-control-identity") is True
    with pytest.raises(RuntimeError, match="old generation state"):
        worker_handle_mod.open_fte_registry_for_query("q-control-identity")

    # A real teardown clears the terminal error after remote storage cleanup.
    # This focused barrier test has no cleanup endpoint, so release its
    # deliberately retained ownership before reopening the global test state.
    with handle._fte_control_lock:
        handle._fte_prepare_terminal_errors.pop("q-control-identity")
    worker_handle_mod.open_fte_registry_for_query("q-control-identity")


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
    actor.fte_prepare_drop_query = _DeferredDrop(_DeferredRef(future))
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
    actor1.fte_prepare_drop_query = _DropMethod(pending_ref)
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
    assert actor1.fte_prepare_drop_query.calls == [query_id]
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


def test_terminal_control_failure_survives_simultaneous_pending_control(monkeypatch):
    class _FailedFuture:
        def result(self, timeout=None):
            raise RuntimeError("planned terminal control failure")

    class _PendingFuture:
        def __init__(self):
            self.value = None

        def result(self, timeout=None):
            if self.value is None:
                raise TimeoutError("planned pending control")
            return self.value

        def complete(self, value):
            self.value = value

    class _Ref:
        def __init__(self, future):
            self._future = future

        def future(self):
            return self._future

    class _AckMethod:
        def __init__(self, failed_ref, pending_ref):
            self.failed_ref = failed_ref
            self.pending_ref = pending_ref

        def remote(self, task_id, *_args):
            if int(task_id["partition_id"]) == 0:
                return self.failed_ref
            return self.pending_ref

    query_id = "query-terminal-and-pending-control"
    failed_task_id = {
        "query_id": query_id,
        "fragment_execution_id": 0,
        "partition_id": 0,
        "attempt_id": 0,
    }
    pending_task_id = {
        "query_id": query_id,
        "fragment_execution_id": 0,
        "partition_id": 1,
        "attempt_id": 0,
    }
    pending_future = _PendingFuture()
    actor = _FakeActor()
    actor.fte_ack_task_result = _AckMethod(_Ref(_FailedFuture()), _Ref(pending_future))
    handle = RayWorkerActorHandle(actor, memory_capacity_bytes=1 << 60)
    handle.enqueue_fte_ack_task_result(failed_task_id)
    handle.enqueue_fte_ack_task_result(pending_task_id)

    def resolve(refs, **_kwargs):
        if isinstance(refs, list):
            return [ref.future().result() for ref in refs]
        return refs.future().result()

    monkeypatch.setattr(
        task_control_mod,
        "resolve_object_refs_blocking",
        resolve,
    )

    with pytest.raises(
        task_control_mod.FteControlBarrierPendingError,
        match="retained pending control ownership",
    ):
        handle.fte_drop_query(query_id)

    assert [call for call in actor.fte_calls if call[0] == "drop_query"] == []
    pending_future.complete(
        {
            "state": "FINISHED",
            "task_id": pending_task_id,
            "_fte_control_operation": "fte_ack_task_result",
            "_fte_control_applied": True,
        }
    )

    with pytest.raises(RuntimeError, match="planned terminal control failure"):
        handle.fte_drop_query(query_id)

    assert [call for call in actor.fte_calls if call[0] == "drop_query"] == [("drop_query", query_id)]
    assert [call for call in actor.fte_calls if call[0] == "cleanup_query"] == [("cleanup_query", query_id)]
    assert handle._has_fte_control_state_for_query(query_id) is False
    assert handle._has_fte_teardown_state_for_query(query_id) is False
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


def test_terminal_control_failure_survives_retryable_remote_drop_failure(monkeypatch):
    monkeypatch.setenv("VANE_FTE_CONTROL_RPC_MAX_ATTEMPTS", "1")

    class _FailedFuture:
        def result(self, timeout=None):
            raise RuntimeError("planned terminal control failure")

    class _Ref:
        def future(self):
            return _FailedFuture()

    class _AckMethod:
        def remote(self, *_args):
            return _Ref()

    class _FailOnceActor(_FakeActor):
        def __init__(self):
            super().__init__()
            self.fail_once = True

        def _fte_drop_query(self, query_id):
            self.fte_calls.append(("drop_query", query_id))
            if self.fail_once:
                self.fail_once = False
                raise RuntimeError("planned remote drop failure")
            return {
                "tasks_removed": 1,
                "tasks_canceled": 0,
                "fragments_removed": 2,
            }

    query_id = "query-terminal-control-with-retryable-drop"
    task_id = {
        "query_id": query_id,
        "fragment_execution_id": 0,
        "partition_id": 0,
        "attempt_id": 0,
    }
    actor = _FailOnceActor()
    actor.fte_ack_task_result = _AckMethod()
    handle = RayWorkerActorHandle(actor, memory_capacity_bytes=1 << 60)
    handle.enqueue_fte_ack_task_result(task_id)

    def resolve(refs, **_kwargs):
        if isinstance(refs, list):
            raise RuntimeError("planned terminal control failure")
        return refs.future().result()

    monkeypatch.setattr(
        task_control_mod,
        "resolve_object_refs_blocking",
        resolve,
    )

    with pytest.raises(RuntimeError, match="planned remote drop failure"):
        handle.fte_drop_query(query_id)

    assert handle._has_fte_teardown_state_for_query(query_id) is True

    with pytest.raises(RuntimeError, match="planned terminal control failure"):
        handle.fte_drop_query(query_id)

    assert [call for call in actor.fte_calls if call[0] == "drop_query"] == [
        ("drop_query", query_id),
        ("drop_query", query_id),
    ]
    assert [call for call in actor.fte_calls if call[0] == "cleanup_query"] == [("cleanup_query", query_id)]
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


def test_control_barrier_ignores_terminal_error_from_detached_ref(monkeypatch):
    query_id = "query-control-detached-terminal-error"
    task_id = {
        "query_id": query_id,
        "fragment_execution_id": 0,
        "partition_id": 0,
        "attempt_id": 0,
    }
    raw_status = {
        "state": "FINISHED",
        "task_id": task_id,
        "_fte_control_operation": "fte_ack_task_result",
        "_fte_control_applied": True,
    }
    actor = _FakeActor()
    handle = RayWorkerActorHandle(actor, memory_capacity_bytes=1 << 60)
    ref = handle.enqueue_fte_ack_task_result(task_id)
    validation_failed = threading.Event()
    current_ref_confirmed = threading.Event()
    allow_error_recording = threading.Event()
    original_is_current = handle._is_current_ordered_fte_control_ref

    def resolve(refs, **_kwargs):
        assert refs == [ref]
        return [raw_status]

    def fail_validation(_status, _expected_task):
        validation_failed.set()
        raise RuntimeError("canceled detached ref")

    def gate_after_current_ref_check(task_key, candidate_ref):
        is_current = original_is_current(task_key, candidate_ref)
        if validation_failed.is_set() and is_current and not current_ref_confirmed.is_set():
            current_ref_confirmed.set()
            assert allow_error_recording.wait(timeout=1.0)
        return is_current

    monkeypatch.setattr(task_control_mod, "resolve_object_refs_blocking", resolve)
    monkeypatch.setattr(task_control_mod, "validate_fte_status_identity", fail_validation)
    monkeypatch.setattr(handle, "_is_current_ordered_fte_control_ref", gate_after_current_ref_check)

    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            closed = executor.submit(handle.close_and_flush_fte_controls, query_id)
            assert current_ref_confirmed.wait(timeout=1.0)
            handle._discard_ordered_fte_control_ref(task_id, ref)
            allow_error_recording.set()
            assert closed.result(timeout=1.0) == []

        assert handle._has_fte_control_state_for_query(query_id) is False
        assert query_id not in handle._fte_prepare_terminal_errors
    finally:
        allow_error_recording.set()
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


@pytest.mark.parametrize(
    ("invalid_response", "error_pattern"),
    [
        pytest.param(
            {
                "state": FteTaskState.RUNNING.value,
                "_fte_control_operation": "fte_add_splits",
                "_fte_control_applied": False,
            },
            "not applied to non-terminal task",
            id="non-terminal",
        ),
        pytest.param(
            {
                "state": FteTaskState.FAILED.value,
                "_fte_control_operation": "fte_update_task",
                "_fte_control_applied": False,
            },
            "control operation mismatch",
            id="operation-mismatch",
        ),
        pytest.param(
            {
                "state": FteTaskState.FAILED.value,
                "_fte_control_operation": "fte_add_splits",
                "_fte_control_applied": False,
                "partition_id": 2,
            },
            "status identity mismatch",
            id="identity-mismatch",
        ),
    ],
)
def test_fte_worker_actor_handle_rejects_invalid_unapplied_add_status(
    monkeypatch,
    invalid_response,
    error_pattern,
):
    task_id = {
        "query_id": "query-invalid-unapplied-add",
        "fragment_execution_id": 0,
        "partition_id": 1,
        "attempt_id": 0,
    }
    status = {
        **invalid_response,
        "task_id": {
            **task_id,
            "partition_id": invalid_response.get("partition_id", task_id["partition_id"]),
        },
    }
    handle = RayWorkerActorHandle(
        _FakeActor(),
        memory_capacity_bytes=1 << 60,
        worker_id="worker-invalid-unapplied-add",
    )
    monkeypatch.setattr(
        handle,
        "_enqueue_ordered_fte_control_rpc",
        lambda *_args, **_kwargs: status,
    )

    with pytest.raises(RuntimeError, match=error_pattern):
        handle.enqueue_fte_add_splits(
            task_id,
            "7",
            [{"sequence_id": 1}],
        )


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
    handle0.record_fte_task_terminal(
        {
            "query_id": "query-drop",
            "fragment_execution_id": 99,
            "partition_id": 0,
            "attempt_id": 0,
        },
        drain=False,
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
    assert sum(worker["terminal_attempt_count"] for worker in before["workers"].values()) == 1

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
    assert sum(worker["terminal_attempt_count"] for worker in after["workers"].values()) == 0
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
                "cleanup_pending": 0,
                "active_executions": 0,
                "last_error": "",
            }

        return _cleanup

    monkeypatch.setattr(worker_mod, "require_ray_cxx_attr", _fake_require)

    result = worker_mod._cleanup_flight_shuffle_for_query("query-drop")

    assert result == {
        "registry_entries_removed": 2,
        "storage_entries_removed": 7,
        "cleanup_errors": 0,
        "cleanup_storage_required": 0,
        "cleanup_pending": 0,
        "active_executions": 0,
        "last_error": "",
    }
    assert calls == [("query-drop", "Ensure the C++ ray extension is built with Flight shuffle cleanup support.")]


def test_worker_flight_shuffle_cleanup_helper_passes_cleanup_connection_and_snapshot(monkeypatch):
    calls = []
    cleanup_connection = object()

    def _fake_require(name, hint=None):
        assert name == "cleanup_flight_shuffle_for_query"

        def _cleanup(
            query_id,
            connection,
            snapshot_query_id,
            apply_snapshot_s3_credentials,
            effective_session_config,
        ):
            calls.append(
                (
                    query_id,
                    connection,
                    snapshot_query_id,
                    apply_snapshot_s3_credentials,
                    effective_session_config,
                    hint,
                )
            )
            return {
                "registry_entries_removed": 1,
                "storage_entries_removed": 2,
                "cleanup_errors": 0,
                "cleanup_pending": 0,
                "active_executions": 0,
                "last_error": "",
            }

        return _cleanup

    monkeypatch.setattr(worker_mod, "require_ray_cxx_attr", _fake_require)

    result = worker_mod._cleanup_flight_shuffle_for_query(
        "query-drop",
        cleanup_connection,
        "resource-query",
        apply_snapshot_s3_credentials=False,
        effective_session_config={"AWS_REGION": "region-a"},
    )

    assert result["storage_entries_removed"] == 2
    assert calls == [
        (
            "query-drop",
            cleanup_connection,
            "resource-query",
            False,
            {"AWS_REGION": "region-a"},
            "Ensure the C++ ray extension is built with Flight shuffle cleanup support.",
        )
    ]


def test_worker_local_shuffle_cleanup_skips_object_storage_rebuild(monkeypatch):
    actor_class = worker_mod.RayWorkerActor.__ray_metadata__.modified_class
    actor = object.__new__(actor_class)
    actor._native_execution_condition = threading.Condition()
    actor._native_query_cleanup_contexts = {
        "query-drop": worker_mod.NativeQueryCleanupContext(
            session_id="session-a",
            session_config=(("AWS_PROFILE", "unavailable-profile"),),
            use_session_credentials=True,
            connection_snapshot_query_id="resource-query",
            connection_snapshot_identity=worker_mod.CleanupConnectionSnapshotIdentity(":memory:", False, (), ()),
        )
    }
    cleanup = {
        "registry_entries_removed": 1,
        "storage_entries_removed": 2,
        "cleanup_errors": 0,
        "cleanup_storage_required": 0,
        "cleanup_pending": 0,
        "active_executions": 0,
        "last_error": "",
    }
    monkeypatch.setattr(
        worker_mod,
        "_cleanup_flight_shuffle_for_query",
        lambda query_id, connection=None, connection_snapshot_query_id="": cleanup,
    )
    monkeypatch.setattr(
        worker_mod,
        "_refresh_effective_duckdb_s3_config",
        lambda *args, **kwargs: pytest.fail("local cleanup must not refresh AWS credentials"),
    )

    result = actor_class._cleanup_flight_shuffle_for_query_with_context(actor, "query-drop")

    assert result == cleanup


def test_worker_object_shuffle_cleanup_uses_refreshed_dedicated_cursor(monkeypatch):
    actor_class = worker_mod.RayWorkerActor.__ray_metadata__.modified_class
    actor = object.__new__(actor_class)
    actor._native_execution_condition = threading.Condition()
    actor._session_connections_lock = threading.RLock()
    actor._session_connections = {"session-a": ({}, object())}
    actor._session_s3_configs = {
        "session-a": {
            "AWS_ACCESS_KEY_ID": "stale-key",
            "AWS_SECRET_ACCESS_KEY": "stale-secret",
            worker_mod._AWS_CREDENTIAL_REFRESH_AT_KEY: "0",
        }
    }
    actor._native_query_cleanup_contexts = {
        "query-drop": worker_mod.NativeQueryCleanupContext(
            session_id="session-a",
            session_config=(("AWS_PROFILE", "profile-a"),),
            use_session_credentials=True,
            connection_snapshot_query_id="resource-query",
            connection_snapshot_identity=worker_mod.CleanupConnectionSnapshotIdentity(":memory:", False, (), ()),
        )
    }
    events = []

    class _CleanupCursor:
        def close(self):
            events.append(("close",))

    cleanup_cursor = _CleanupCursor()
    actor._get_shared_conn = lambda: SimpleNamespace(cursor=lambda: cleanup_cursor)

    def refresh(config, cached, *, use_session_credentials):
        events.append(("refresh", dict(config), dict(cached), use_session_credentials))
        return {
            "AWS_ACCESS_KEY_ID": "fresh-key",
            "AWS_SECRET_ACCESS_KEY": "fresh-secret",
            worker_mod._AWS_CREDENTIAL_REFRESH_AT_KEY: "9999999999",
        }

    def configure(connection, config, *, use_session_credentials):
        events.append(("configure", connection, dict(config), use_session_credentials))
        return dict(config)

    def cleanup(
        query_id,
        connection=None,
        connection_snapshot_query_id="",
        *,
        apply_snapshot_s3_credentials=True,
        effective_session_config=None,
    ):
        if connection is None:
            events.append(("probe", query_id))
            return {
                "registry_entries_removed": 1,
                "storage_entries_removed": 0,
                "cleanup_errors": 1,
                "cleanup_storage_required": 1,
                "cleanup_pending": 1,
                "active_executions": 0,
                "last_error": "shuffle cleanup requires a live filesystem context",
            }
        events.append(
            (
                "cleanup",
                query_id,
                connection,
                connection_snapshot_query_id,
                apply_snapshot_s3_credentials,
                effective_session_config,
            )
        )
        return {
            "registry_entries_removed": 0,
            "storage_entries_removed": 2,
            "cleanup_errors": 0,
            "cleanup_storage_required": 0,
            "cleanup_pending": 0,
            "active_executions": 0,
            "last_error": "",
        }

    monkeypatch.setattr(worker_mod, "_refresh_effective_duckdb_s3_config", refresh)
    monkeypatch.setattr(worker_mod, "_configure_duckdb_s3", configure)
    monkeypatch.setattr(worker_mod, "_cleanup_flight_shuffle_for_query", cleanup)

    result = actor_class._cleanup_flight_shuffle_for_query_with_context(actor, "query-drop")

    assert result == {
        "registry_entries_removed": 1,
        "storage_entries_removed": 2,
        "cleanup_errors": 0,
        "cleanup_storage_required": 0,
        "cleanup_pending": 0,
        "active_executions": 0,
        "last_error": "",
    }
    assert actor._session_s3_configs["session-a"]["AWS_ACCESS_KEY_ID"] == "fresh-key"
    assert events == [
        ("probe", "query-drop"),
        (
            "refresh",
            {"AWS_PROFILE": "profile-a"},
            {
                "AWS_ACCESS_KEY_ID": "stale-key",
                "AWS_SECRET_ACCESS_KEY": "stale-secret",
                worker_mod._AWS_CREDENTIAL_REFRESH_AT_KEY: "0",
            },
            True,
        ),
        (
            "configure",
            cleanup_cursor,
            {
                "AWS_ACCESS_KEY_ID": "fresh-key",
                "AWS_SECRET_ACCESS_KEY": "fresh-secret",
                worker_mod._AWS_CREDENTIAL_REFRESH_AT_KEY: "9999999999",
            },
            True,
        ),
        (
            "cleanup",
            "query-drop",
            cleanup_cursor,
            "resource-query",
            False,
            {
                "AWS_ACCESS_KEY_ID": "fresh-key",
                "AWS_SECRET_ACCESS_KEY": "fresh-secret",
                worker_mod._AWS_CREDENTIAL_REFRESH_AT_KEY: "9999999999",
            },
        ),
        ("close",),
    ]


def test_worker_object_shuffle_cleanup_replays_explicit_connection_snapshot(monkeypatch):
    actor_class = worker_mod.RayWorkerActor.__ray_metadata__.modified_class
    actor = object.__new__(actor_class)
    actor._native_execution_condition = threading.Condition()
    actor._session_connections_lock = threading.RLock()
    actor._session_connections = {}
    actor._session_s3_configs = {}
    actor._native_query_cleanup_contexts = {
        "query-drop": worker_mod.NativeQueryCleanupContext(
            session_id="session-a",
            session_config=(("AWS_ENDPOINT_URL", "https://s3.example.test"),),
            use_session_credentials=False,
            connection_snapshot_query_id="resource-query",
            connection_snapshot_identity=worker_mod.CleanupConnectionSnapshotIdentity(":memory:", False, (), ()),
        )
    }
    cleanup_cursor = SimpleNamespace(close=lambda: None)
    actor._get_shared_conn = lambda: SimpleNamespace(cursor=lambda: cleanup_cursor)
    cleanup_calls = []

    def cleanup(
        query_id,
        connection=None,
        connection_snapshot_query_id="",
        *,
        apply_snapshot_s3_credentials=True,
        effective_session_config=None,
    ):
        cleanup_calls.append(
            (
                query_id,
                connection,
                connection_snapshot_query_id,
                apply_snapshot_s3_credentials,
                effective_session_config,
            )
        )
        if connection is None:
            return {
                "registry_entries_removed": 1,
                "storage_entries_removed": 0,
                "cleanup_errors": 1,
                "cleanup_storage_required": 1,
                "cleanup_pending": 1,
                "active_executions": 0,
                "last_error": "shuffle cleanup requires a live filesystem context",
            }
        return {
            "registry_entries_removed": 0,
            "storage_entries_removed": 2,
            "cleanup_errors": 0,
            "cleanup_storage_required": 0,
            "cleanup_pending": 0,
            "active_executions": 0,
            "last_error": "",
        }

    monkeypatch.setattr(
        worker_mod,
        "_refresh_effective_duckdb_s3_config",
        lambda config, cached, *, use_session_credentials: dict(config),
    )
    monkeypatch.setattr(
        worker_mod,
        "_configure_duckdb_s3",
        lambda connection, config, *, use_session_credentials: dict(config),
    )
    monkeypatch.setattr(
        worker_mod,
        "_cleanup_flight_shuffle_for_query",
        cleanup,
    )

    result = actor_class._cleanup_flight_shuffle_for_query_with_context(actor, "query-drop")

    assert result == {
        "registry_entries_removed": 1,
        "storage_entries_removed": 2,
        "cleanup_errors": 0,
        "cleanup_storage_required": 0,
        "cleanup_pending": 0,
        "active_executions": 0,
        "last_error": "",
    }
    assert cleanup_calls == [
        ("query-drop", None, "", True, None),
        (
            "query-drop",
            cleanup_cursor,
            "resource-query",
            True,
            {"AWS_ENDPOINT_URL": "https://s3.example.test"},
        ),
    ]


def test_worker_object_shuffle_cleanup_preserves_primary_error_when_cursor_close_fails(monkeypatch):
    actor_class = worker_mod.RayWorkerActor.__ray_metadata__.modified_class
    actor = object.__new__(actor_class)
    actor._native_execution_condition = threading.Condition()
    actor._session_connections_lock = threading.RLock()
    actor._session_connections = {}
    actor._session_s3_configs = {}
    actor._native_query_cleanup_contexts = {
        "query-drop": worker_mod.NativeQueryCleanupContext(
            session_id="session-a",
            session_config=(),
            use_session_credentials=True,
            connection_snapshot_query_id="resource-query",
            connection_snapshot_identity=worker_mod.CleanupConnectionSnapshotIdentity(":memory:", False, (), ()),
        )
    }

    class _CleanupCursor:
        def close(self):
            raise RuntimeError("cursor close failed")

    actor._get_shared_conn = lambda: SimpleNamespace(cursor=_CleanupCursor)
    monkeypatch.setattr(
        worker_mod,
        "_cleanup_flight_shuffle_for_query",
        lambda query_id, connection=None, connection_snapshot_query_id="": {
            "registry_entries_removed": 1,
            "storage_entries_removed": 0,
            "cleanup_errors": 1,
            "cleanup_storage_required": 1,
            "cleanup_pending": 1,
            "active_executions": 0,
            "last_error": "shuffle cleanup requires a live filesystem context",
        },
    )
    monkeypatch.setattr(
        worker_mod,
        "_refresh_effective_duckdb_s3_config",
        lambda config, cached, *, use_session_credentials: {},
    )
    monkeypatch.setattr(
        worker_mod,
        "_configure_duckdb_s3",
        lambda connection, config, *, use_session_credentials: (_ for _ in ()).throw(
            RuntimeError("cleanup configuration failed")
        ),
    )

    with pytest.raises(RuntimeError, match="cleanup configuration failed"):
        actor_class._cleanup_flight_shuffle_for_query_with_context(actor, "query-drop")


def test_worker_cleanup_context_replays_snapshot_with_session_credentials(monkeypatch):
    actor_class = worker_mod.RayWorkerActor.__ray_metadata__.modified_class
    actor = object.__new__(actor_class)
    actor._native_execution_condition = threading.Condition()
    default_identity = worker_mod.CleanupConnectionSnapshotIdentity(":memory:", False, (), ())
    monkeypatch.setattr(worker_mod, "_query_cleanup_connection_identity", lambda *args, **kwargs: default_identity)

    class _Plan:
        @staticmethod
        def resource_query_id():
            return "resource-query"

    actor_class._register_native_query_cleanup_context(
        actor,
        "execution-query",
        _Plan(),
        "session-a",
        {"AWS_PROFILE": "profile-a"},
        use_session_credentials=True,
    )

    assert actor._native_query_cleanup_contexts == {
        "execution-query": worker_mod.NativeQueryCleanupContext(
            session_id="session-a",
            session_config=(("AWS_PROFILE", "profile-a"),),
            use_session_credentials=True,
            connection_snapshot_query_id="resource-query",
            connection_snapshot_identity=default_identity,
        )
    }


def test_worker_cleanup_connection_identity_matches_snapshot_replay(monkeypatch):
    snapshot = {
        "bootstrap": {
            "database": ":memory:",
            "read_only": False,
            "config": {
                "s3_endpoint": "bootstrap.example.test",
                "s3_region": "bootstrap-region",
            },
        },
        "settings": [
            {"name": "s3_endpoint", "value": "s3.example.test", "input_type": "VARCHAR"},
            {"name": "s3_access_key_id", "value": "plan-key", "input_type": "VARCHAR"},
            {"name": "s3_secret_access_key", "value": "plan-secret", "input_type": "VARCHAR"},
            {"name": "http_timeout", "value": "30", "input_type": "BIGINT"},
            {"name": "allow_unsigned_extensions", "value": "true", "input_type": "BOOLEAN"},
        ],
    }

    def require(name, *, hint):
        assert name == "_lookup_query_connection_snapshot"
        assert hint == "Ensure the C++ ray extension is built with query replay lifecycle support."
        return lambda query_id: snapshot if query_id == "resource-query" else None

    monkeypatch.setattr(worker_mod, "require_ray_cxx_attr", require)

    assert worker_mod._query_cleanup_connection_identity(
        "resource-query",
        apply_snapshot_s3_credentials=False,
    ) == worker_mod.CleanupConnectionSnapshotIdentity(
        bootstrap_database=":memory:",
        bootstrap_read_only=False,
        bootstrap_config=(
            ("s3_endpoint", "bootstrap.example.test"),
            ("s3_region", "bootstrap-region"),
        ),
        settings=(
            ("s3_endpoint", "s3.example.test", "VARCHAR"),
            ("http_timeout", "30", "BIGINT"),
        ),
    )
    assert worker_mod._query_cleanup_connection_identity(
        "resource-query",
        apply_snapshot_s3_credentials=True,
    ) == worker_mod.CleanupConnectionSnapshotIdentity(
        bootstrap_database=":memory:",
        bootstrap_read_only=False,
        bootstrap_config=(
            ("s3_endpoint", "bootstrap.example.test"),
            ("s3_region", "bootstrap-region"),
        ),
        settings=(
            ("s3_endpoint", "s3.example.test", "VARCHAR"),
            ("s3_access_key_id", "plan-key", "VARCHAR"),
            ("s3_secret_access_key", "plan-secret", "VARCHAR"),
            ("http_timeout", "30", "BIGINT"),
        ),
    )


def test_worker_cleanup_context_reuses_first_equivalent_snapshot_across_query_fragments(monkeypatch):
    actor_class = worker_mod.RayWorkerActor.__ray_metadata__.modified_class
    actor = object.__new__(actor_class)
    actor._native_execution_condition = threading.Condition()
    default_identity = worker_mod.CleanupConnectionSnapshotIdentity(
        ":memory:",
        False,
        (("s3_endpoint", "s3.example.test"),),
        (),
    )
    snapshot_identities = {
        "resource-query-a": default_identity,
        "resource-query-b": default_identity,
        "resource-query-c": worker_mod.CleanupConnectionSnapshotIdentity(
            ":memory:",
            False,
            (("s3_endpoint", "other.example.test"),),
            (),
        ),
        "resource-query-d": default_identity,
    }
    monkeypatch.setattr(
        worker_mod,
        "_query_cleanup_connection_identity",
        lambda query_id, **kwargs: snapshot_identities[query_id],
    )

    class _Plan:
        def __init__(self, resource_query_id):
            self._resource_query_id = resource_query_id

        def resource_query_id(self):
            return self._resource_query_id

    actor_class._register_native_query_cleanup_context(
        actor,
        "execution-query",
        _Plan("resource-query-a"),
        "session-a",
        {"AWS_PROFILE": "profile-a"},
        use_session_credentials=True,
    )
    actor_class._register_native_query_cleanup_context(
        actor,
        "execution-query",
        _Plan("resource-query-b"),
        "session-a",
        {"AWS_PROFILE": "profile-a"},
        use_session_credentials=True,
    )

    assert actor._native_query_cleanup_contexts["execution-query"].connection_snapshot_query_id == "resource-query-a"
    with pytest.raises(RuntimeError, match="native query cleanup context changed"):
        actor_class._register_native_query_cleanup_context(
            actor,
            "execution-query",
            _Plan("resource-query-c"),
            "session-a",
            {"AWS_PROFILE": "profile-a"},
            use_session_credentials=True,
        )
    with pytest.raises(RuntimeError, match="native query cleanup context changed"):
        actor_class._register_native_query_cleanup_context(
            actor,
            "execution-query",
            _Plan("resource-query-d"),
            "session-a",
            {"AWS_PROFILE": "profile-b"},
            use_session_credentials=True,
        )


def test_worker_flight_shuffle_cleanup_drain_retries_pending_work(monkeypatch):
    cleanups = iter(
        [
            {
                "registry_entries_removed": 2,
                "storage_entries_removed": 0,
                "cleanup_errors": 1,
                "cleanup_pending": 1,
                "active_executions": 1,
            },
            {
                "registry_entries_removed": 0,
                "storage_entries_removed": 7,
                "cleanup_errors": 0,
                "cleanup_pending": 0,
                "active_executions": 0,
            },
        ]
    )
    monkeypatch.setattr(worker_mod, "_cleanup_flight_shuffle_for_query", lambda _query_id: next(cleanups))

    result = asyncio.run(worker_mod._drain_flight_shuffle_for_query("query-drop", timeout_s=1))

    assert result == {
        "registry_entries_removed": 2,
        "storage_entries_removed": 7,
        "cleanup_errors": 0,
        "cleanup_pending": 0,
        "active_executions": 0,
    }


def test_worker_flight_shuffle_cleanup_timeout_reports_storage_error(monkeypatch):
    monkeypatch.setattr(
        worker_mod,
        "_cleanup_flight_shuffle_for_query",
        lambda _query_id: {
            "registry_entries_removed": 0,
            "storage_entries_removed": 0,
            "cleanup_errors": 1,
            "cleanup_pending": 1,
            "active_executions": 0,
            "last_error": "permission denied",
        },
    )

    with pytest.raises(RuntimeError, match="last_error='permission denied'"):
        asyncio.run(worker_mod._drain_flight_shuffle_for_query("query-drop", timeout_s=0))


def test_worker_drop_query_uses_separate_execution_and_storage_barriers(monkeypatch):
    events: list[str] = []

    class TaskManager:
        async def drop_query(self, query_id):
            assert query_id == "query-drop"
            events.append("cancel")
            return {"removed": 2, "canceled": 1}

    class DummyWorker:
        @staticmethod
        def _get_fte_task_manager():
            return TaskManager()

        @staticmethod
        def drop_query_fragments(query_id):
            assert query_id == "query-drop"
            events.append("fragments")
            return 3

        @staticmethod
        def _close_worker_native_query(query_id):
            assert query_id == "query-drop"
            events.append("native-close")
            return []

        @staticmethod
        async def _wait_worker_native_executions_for_query(query_id):
            assert query_id == "query-drop"
            events.append("worker-wait")

        @staticmethod
        def _retire_worker_native_query(query_id):
            assert query_id == "query-drop"
            events.append("native-retire")

    def close(query_id):
        assert query_id == "query-drop"
        events.append("close")

    async def wait_for_executions(query_id):
        assert query_id == "query-drop"
        events.append("flight-wait")

    async def drain(query_id):
        assert query_id == "query-drop"
        events.append("drain")
        return {
            "registry_entries_removed": 4,
            "storage_entries_removed": 5,
            "cleanup_errors": 0,
            "cleanup_pending": 0,
            "active_executions": 0,
            "last_error": "",
        }

    def retire(query_id):
        assert query_id == "query-drop"
        events.append("retire")

    def release_datasource_factories(query_id):
        assert query_id == "query-drop"
        events.append("datasource-release")
        return 2

    def cleanup_replay(query_id):
        assert query_id == "query-drop"
        events.append("replay-cleanup")

    monkeypatch.setattr(worker_mod, "_close_flight_shuffle_query", close)
    monkeypatch.setattr(worker_mod, "_wait_flight_shuffle_executions_for_query", wait_for_executions)
    monkeypatch.setattr(worker_mod, "_drain_flight_shuffle_for_query", drain)
    monkeypatch.setattr(worker_mod, "_retire_flight_shuffle_query", retire)
    monkeypatch.setattr(worker_mod, "_release_datasource_factories_for_query", release_datasource_factories)
    monkeypatch.setattr(worker_mod, "_cleanup_query_python_replay_state", cleanup_replay)
    actor_class = worker_mod.RayWorkerActor.__ray_metadata__.modified_class

    prepare_result = asyncio.run(actor_class.fte_prepare_drop_query(DummyWorker(), "query-drop"))
    cleanup_result = asyncio.run(actor_class.fte_cleanup_query(DummyWorker(), "query-drop"))
    result = dict(prepare_result)
    result.update(cleanup_result)

    assert events == [
        "close",
        "native-close",
        "cancel",
        "fragments",
        "worker-wait",
        "flight-wait",
        "datasource-release",
        "drain",
        "retire",
        "native-retire",
        "replay-cleanup",
    ]
    assert result == {
        "tasks_removed": 2,
        "tasks_canceled": 1,
        "fragments_removed": 3,
        "flight_shuffle_registry_entries_removed": 4,
        "flight_shuffle_storage_entries_removed": 5,
        "flight_shuffle_cleanup_errors": 0,
    }


def test_worker_drop_query_releases_datasource_after_fragment_cleanup_failure(monkeypatch):
    events: list[str] = []

    class TaskManager:
        async def drop_query(self, query_id):
            assert query_id == "query-drop"
            events.append("cancel")
            return {"removed": 2, "canceled": 1}

    class DummyWorker:
        @staticmethod
        def _get_fte_task_manager():
            return TaskManager()

        @staticmethod
        def drop_query_fragments(query_id):
            assert query_id == "query-drop"
            events.append("fragments")
            raise RuntimeError("fragment cleanup failed")

        @staticmethod
        def _close_worker_native_query(query_id):
            assert query_id == "query-drop"
            events.append("native-close")
            return []

        @staticmethod
        async def _wait_worker_native_executions_for_query(query_id):
            assert query_id == "query-drop"
            events.append("worker-wait")

    monkeypatch.setattr(worker_mod, "_close_flight_shuffle_query", lambda _query_id: events.append("close"))

    async def wait_for_executions(query_id):
        assert query_id == "query-drop"
        events.append("flight-wait")

    monkeypatch.setattr(worker_mod, "_wait_flight_shuffle_executions_for_query", wait_for_executions)
    monkeypatch.setattr(
        worker_mod,
        "_release_datasource_factories_for_query",
        lambda _query_id: events.append("datasource-release"),
    )
    actor_class = worker_mod.RayWorkerActor.__ray_metadata__.modified_class

    with pytest.raises(RuntimeError, match="fragment cleanup failed"):
        asyncio.run(actor_class.fte_prepare_drop_query(DummyWorker(), "query-drop"))

    assert events == [
        "close",
        "native-close",
        "cancel",
        "fragments",
        "worker-wait",
        "flight-wait",
        "datasource-release",
    ]


@pytest.mark.parametrize(
    ("failed_barrier", "expect_release"),
    [
        pytest.param("flight", True, id="flight-failure"),
        pytest.param("native", False, id="native-failure"),
    ],
)
def test_worker_drop_query_releases_datasource_only_after_native_drain(monkeypatch, failed_barrier, expect_release):
    events: list[str] = []

    class TaskManager:
        async def drop_query(self, query_id):
            assert query_id == "query-drop"
            events.append("cancel")
            return {"removed": 2, "canceled": 1}

    class DummyWorker:
        @staticmethod
        def _get_fte_task_manager():
            return TaskManager()

        @staticmethod
        def drop_query_fragments(query_id):
            assert query_id == "query-drop"
            events.append("fragments")
            return 3

        @staticmethod
        def _close_worker_native_query(query_id):
            assert query_id == "query-drop"
            events.append("native-close")
            return []

        @staticmethod
        async def _wait_worker_native_executions_for_query(query_id):
            assert query_id == "query-drop"
            events.append("worker-wait")
            if failed_barrier == "native":
                raise RuntimeError("native drain failed")

    async def wait_for_flight(query_id):
        assert query_id == "query-drop"
        events.append("flight-wait")
        if failed_barrier == "flight":
            raise RuntimeError("flight drain failed")

    monkeypatch.setattr(worker_mod, "_close_flight_shuffle_query", lambda _query_id: events.append("close"))
    monkeypatch.setattr(worker_mod, "_wait_flight_shuffle_executions_for_query", wait_for_flight)
    monkeypatch.setattr(
        worker_mod,
        "_release_datasource_factories_for_query",
        lambda _query_id: events.append("datasource-release"),
    )
    actor_class = worker_mod.RayWorkerActor.__ray_metadata__.modified_class

    with pytest.raises(RuntimeError, match=f"{failed_barrier} drain failed"):
        asyncio.run(actor_class.fte_prepare_drop_query(DummyWorker(), "query-drop"))

    expected_events = [
        "close",
        "native-close",
        "cancel",
        "fragments",
        "worker-wait",
        "flight-wait",
    ]
    if expect_release:
        expected_events.append("datasource-release")
    assert events == expected_events


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


def test_fte_terminal_pressure_rejects_late_attempt_updates():
    handle = RayWorkerActorHandle(
        _FakeActor(),
        memory_capacity_bytes=1 << 60,
        worker_id="worker-terminal-pressure",
    )
    attempt = {
        "query_id": "query-terminal-pressure",
        "fragment_execution_id": 0,
        "partition_id": 0,
        "attempt_id": 0,
    }
    request = {
        "execution_class": "standard",
        "initial_splits": {},
        "memory_requirement_bytes": 64,
    }

    assert handle.record_fte_task_started(attempt, request) is True
    assert handle.record_fte_splits_added(attempt, 1, split_bytes=16) is True
    handle.record_fte_task_terminal(attempt, drain=False)

    assert handle.record_fte_task_started(attempt, request) is False
    assert handle.record_fte_splits_added(attempt, 1, split_bytes=128) is False
    assert handle.record_fte_split_bytes_added(attempt, 256) is False
    stats = handle.fte_pressure_stats()
    assert stats["running_attempt_count"] == 0
    assert stats["assigned_split_count"] == 0
    assert stats["assigned_split_bytes"] == 0
    assert stats["assigned_memory_bytes"] == 0
    assert stats["score"] == 0


def test_fte_split_ack_before_create_ack_is_merged_into_running_pressure(monkeypatch):
    handle = RayWorkerActorHandle(
        _FakeActor(),
        memory_capacity_bytes=1 << 60,
        worker_id="worker-create-add-ack-order",
    )
    query_id = "query-create-add-ack-order"
    fragment_id = f"{query_id}:node:7"
    stage = FteFragmentExecution(
        query_id,
        0,
        fragment_id=fragment_id,
        task_memory_bytes=64,
    )
    partition = stage.add_partition(0)
    scheduled = partition.start_attempt(
        worker_id=handle.worker_id,
        worker_incarnation_id=handle.worker_incarnation_id,
        remote_handle=handle,
    )
    request = scheduled.request
    request["initial_splits"] = {
        "7": ({"sequence_id": 0, "kind": "scan_task", "data": b"base"},),
    }
    handle.reserve_fte_partition(
        query_id,
        fragment_id,
        0,
        memory_requirement_bytes=64,
        execution_class="standard",
    )
    create_command = FteCreateTaskCommand(
        query_id=query_id,
        fragment_id=fragment_id,
        worker_id=handle.worker_id,
        worker=handle,
        attempt_id=scheduled.attempt_id,
        partition_id=0,
        request=request,
        scheduled_attempt=scheduled,
    )
    add_command = FteAddSplitsCommand(
        query_id=query_id,
        fragment_id=fragment_id,
        worker_id=handle.worker_id,
        worker=handle,
        attempt_id=scheduled.attempt_id,
        source_node_id="7",
        splits=(
            {"sequence_id": 1, "kind": "scan_task", "data": b"a"},
            {"sequence_id": 2, "kind": "scan_task", "data": b"bc"},
        ),
    )
    create_rpc_completed = threading.Event()
    release_create_ack = threading.Event()
    scheduler = worker_handle_mod._FTE_SCHEDULERS.get_or_create(query_id)

    def execute(command, **_kwargs):
        if isinstance(command, FteCreateTaskCommand):
            create_rpc_completed.set()
            assert release_create_ack.wait(2.0)
        else:
            assert create_rpc_completed.wait(2.0)

    monkeypatch.setattr(scheduler.worker_command_executor, "execute", execute)
    monkeypatch.setattr(
        worker_commands_mod,
        "fte_partition_task_lease_payload",
        lambda *_args, **_kwargs: {},
    )

    with ThreadPoolExecutor(max_workers=2) as executor:
        create_ack = executor.submit(
            handle._execute_fte_fragment_execution_worker_commands,
            stage,
            [create_command],
        )
        assert create_rpc_completed.wait(2.0)
        add_ack = executor.submit(
            handle._execute_fte_fragment_execution_worker_commands,
            stage,
            [add_command],
        )
        add_ack.result(timeout=2.0)

        attempt_key = str(scheduled.attempt_id)
        assert handle._fte_pressure.pending_split_counts_by_attempt == {attempt_key: 2}
        pending_split_bytes = handle._fte_pressure.pending_split_bytes_by_attempt[attempt_key]
        assert pending_split_bytes > 0
        assert handle.fte_pressure_stats()["running_attempt_count"] == 0

        release_create_ack.set()
        create_ack.result(timeout=2.0)

    stats = handle.fte_pressure_stats()
    assert stats["running_attempt_count"] == 1
    assert stats["reserved_partition_count"] == 0
    assert stats["assigned_split_count"] == 3
    assert stats["assigned_split_bytes"] == 4 + pending_split_bytes
    assert stats["assigned_memory_bytes"] == 64
    assert handle._fte_pressure.pending_split_counts_by_attempt == {}
    assert handle._fte_pressure.pending_split_bytes_by_attempt == {}


def test_fte_terminal_pressure_compacts_retry_tombstones_by_task():
    handle = RayWorkerActorHandle(
        _FakeActor(),
        memory_capacity_bytes=1 << 60,
        worker_id="worker-terminal-pressure-compaction",
    )
    task = {
        "query_id": "query-terminal-pressure-compaction",
        "fragment_execution_id": 0,
        "partition_id": 0,
    }
    request = {
        "execution_class": "standard",
        "initial_splits": {},
        "memory_requirement_bytes": 64,
    }

    for attempt_id in range(10_000):
        handle.record_fte_task_terminal(
            {**task, "attempt_id": attempt_id},
            drain=False,
        )

    task_key = "query-terminal-pressure-compaction.0.0"
    assert handle._fte_pressure.terminal_attempt_id_by_task == {task_key: 9_999}
    assert handle.fte_pressure_stats()["terminal_attempt_count"] == 1
    assert handle.record_fte_task_started({**task, "attempt_id": 9_999}, request) is False
    assert handle.record_fte_splits_added({**task, "attempt_id": 9_999}, 1) is False
    assert handle.record_fte_task_started({**task, "attempt_id": 10_000}, request) is True


def test_fte_late_ack_for_terminal_attempt_does_not_touch_retry_pressure():
    handle = RayWorkerActorHandle(
        _FakeActor(),
        memory_capacity_bytes=1 << 60,
        worker_id="worker-retry-pressure",
    )
    old_attempt = {
        "query_id": "query-retry-pressure",
        "fragment_execution_id": 0,
        "partition_id": 0,
        "attempt_id": 0,
    }
    retry_attempt = {
        **old_attempt,
        "attempt_id": 1,
    }
    request = {
        "execution_class": "standard",
        "initial_splits": {},
        "memory_requirement_bytes": 64,
    }

    assert handle.record_fte_task_started(old_attempt, request) is True
    handle.record_fte_task_terminal(old_attempt, drain=False)
    assert handle.record_fte_task_started(retry_attempt, request) is True
    assert handle.record_fte_splits_added(retry_attempt, 2, split_bytes=32) is True

    assert handle.record_fte_splits_added(old_attempt, 10, split_bytes=1024) is False
    stats = handle.fte_pressure_stats()
    assert stats["running_attempt_count"] == 1
    assert stats["assigned_split_count"] == 2
    assert stats["assigned_split_bytes"] == 32
    assert stats["assigned_memory_bytes"] == 64


def test_fte_late_create_ack_does_not_clear_retry_reservation():
    handle = RayWorkerActorHandle(
        _FakeActor(),
        memory_capacity_bytes=1 << 60,
        worker_id="worker-retry-reservation",
    )
    query_id = "query-retry-reservation"
    fragment_id = f"{query_id}:node:7"
    old_attempt = {
        "query_id": query_id,
        "fragment_execution_id": 0,
        "partition_id": 0,
        "attempt_id": 0,
    }
    request = {
        "execution_class": "standard",
        "initial_splits": {},
        "memory_requirement_bytes": 128,
    }

    assert handle.record_fte_task_started(old_attempt, request) is True
    handle.record_fte_task_terminal(old_attempt, drain=False)
    handle.reserve_fte_partition(
        query_id,
        fragment_id,
        0,
        memory_requirement_bytes=128,
        execution_class="standard",
    )

    assert (
        handle.record_fte_task_started_from_reservation(
            query_id,
            fragment_id,
            0,
            old_attempt,
            request,
        )
        is False
    )
    stats = handle.fte_pressure_stats()
    assert stats["running_attempt_count"] == 0
    assert stats["terminal_attempt_count"] == 1
    assert stats["reserved_partition_count"] == 1
    assert stats["reserved_memory_bytes"] == 128
    assert stats["score"] == 1024


def test_fte_add_splits_ack_after_task_finish_does_not_revive_pressure():
    rpc_started = threading.Event()
    release_rpc = threading.Event()

    class _BlockingFuture:
        def __init__(self, value):
            self.value = value
            self._lock = threading.Lock()
            self._callbacks = []
            self._done = False

        def result(self, timeout=None):
            with self._lock:
                if self._done:
                    return self.value
            if not release_rpc.wait(timeout):
                raise FutureTimeoutError
            with self._lock:
                if self._done:
                    return self.value
                self._done = True
                callbacks = list(self._callbacks)
                self._callbacks.clear()
            for callback in callbacks:
                callback(self)
            return self.value

        def add_done_callback(self, callback):
            with self._lock:
                if self._done:
                    invoke_now = True
                else:
                    self._callbacks.append(callback)
                    invoke_now = False
            if invoke_now:
                callback(self)

        def done(self):
            with self._lock:
                return self._done

    class _BlockingObjectRef:
        def __init__(self, value):
            self._future = _BlockingFuture(value)

        def future(self):
            return self._future

    class _BlockingAddSplits:
        def remote(self, task_id, source_node_id, splits, dependency=None):
            rpc_started.set()
            return _BlockingObjectRef(
                _FakeActor._control_status(
                    "fte_add_splits",
                    task_id,
                    version=2,
                )
            )

    query_id = "query-late-add-splits-ack"
    fragment_id = f"{query_id}:node:7"
    actor = _FakeActor()
    actor.fte_add_splits = _BlockingAddSplits()
    handle = RayWorkerActorHandle(
        actor,
        memory_capacity_bytes=1 << 60,
        worker_id="worker-late-add-splits-ack",
    )
    stage = FteFragmentExecution(
        query_id,
        0,
        fragment_id=fragment_id,
        task_memory_bytes=64,
    )

    finished_partition = stage.add_partition(0)
    finished_attempt = finished_partition.start_attempt(
        worker_id=handle.worker_id,
        worker_incarnation_id=handle.worker_incarnation_id,
        remote_handle=handle,
    )
    assert handle.record_fte_task_started(finished_attempt.attempt_id, finished_attempt.request) is True

    retry_partition = stage.add_partition(1)
    failed_attempt = retry_partition.start_attempt(
        worker_id=handle.worker_id,
        worker_incarnation_id=handle.worker_incarnation_id,
        remote_handle=handle,
    )
    assert handle.record_fte_task_started(failed_attempt.attempt_id, failed_attempt.request) is True
    assert (
        stage.task_failed(
            failed_attempt.attempt_id,
            {
                "error_code": "GENERIC_INTERNAL_ERROR",
                "message": "retry",
            },
            schedule_retry=False,
        )
        is None
    )
    retry_attempt = retry_partition.start_attempt(
        worker_id=handle.worker_id,
        worker_incarnation_id=handle.worker_incarnation_id,
        remote_handle=handle,
    )
    assert retry_attempt.attempt_id.attempt_id == 1
    assert handle.record_fte_task_started(retry_attempt.attempt_id, retry_attempt.request) is True
    assert handle.record_fte_splits_added(retry_attempt.attempt_id, 2, split_bytes=32) is True

    command = FteAddSplitsCommand(
        query_id=query_id,
        fragment_id=fragment_id,
        worker_id=handle.worker_id,
        worker=handle,
        attempt_id=finished_attempt.attempt_id,
        source_node_id="7",
        splits=({"sequence_id": 1, "kind": "scan_task", "data": b"late"},),
    )

    with ThreadPoolExecutor(max_workers=1) as executor:
        command_execution = executor.submit(
            handle._execute_fte_fragment_execution_worker_commands,
            stage,
            [command],
        )
        try:
            assert rpc_started.wait(2.0)
            assert stage.task_finished(finished_attempt.attempt_id) is True

            before_ack = handle.fte_pressure_stats()
            assert before_ack["running_attempt_count"] == 1
            assert before_ack["terminal_attempt_count"] == 2
            assert before_ack["assigned_split_count"] == 2
            assert before_ack["assigned_split_bytes"] == 32
            assert before_ack["assigned_memory_bytes"] == 64
        finally:
            release_rpc.set()
        command_execution.result(timeout=2.0)

    after_ack = handle.fte_pressure_stats()
    assert after_ack["running_attempt_count"] == 1
    assert after_ack["terminal_attempt_count"] == 2
    assert after_ack["assigned_split_count"] == 2
    assert after_ack["assigned_split_bytes"] == 32
    assert after_ack["assigned_memory_bytes"] == 64
    assert handle._fte_pressure.running_attempts == {str(retry_attempt.attempt_id)}


def test_fte_pressure_drop_query_serializes_with_other_query_start():
    handle = RayWorkerActorHandle(
        _FakeActor(),
        memory_capacity_bytes=1 << 60,
        worker_id="worker-drop-pressure-race",
    )
    query_a_attempt = {
        "query_id": "query-drop-pressure-a",
        "fragment_execution_id": 0,
        "partition_id": 0,
        "attempt_id": 0,
    }
    query_b_attempt = {
        "query_id": "query-drop-pressure-b",
        "fragment_execution_id": 0,
        "partition_id": 0,
        "attempt_id": 0,
    }
    request = {
        "execution_class": "standard",
        "initial_splits": {},
        "memory_requirement_bytes": 64,
    }
    assert handle.record_fte_task_started(query_a_attempt, request) is True

    drop_iteration_started = threading.Event()
    query_b_start_entered = threading.Event()

    class _BlockingAttemptSet(set):
        def __iter__(self):
            snapshot = tuple(set.__iter__(self))
            drop_iteration_started.set()
            assert query_b_start_entered.wait(2.0)
            return iter(snapshot)

    handle._fte_pressure.running_attempts = _BlockingAttemptSet(handle._fte_pressure.running_attempts)

    def start_query_b():
        query_b_start_entered.set()
        return handle.record_fte_task_started(query_b_attempt, request)

    with ThreadPoolExecutor(max_workers=2) as executor:
        dropped = executor.submit(handle._fte_pressure.drop_query, "query-drop-pressure-a")
        assert drop_iteration_started.wait(2.0)
        started = executor.submit(start_query_b)
        dropped.result(timeout=2.0)
        assert started.result(timeout=2.0) is True

    stats = handle.fte_pressure_stats()
    assert stats["running_attempt_count"] == 1
    assert stats["assigned_memory_bytes"] == 64
    assert handle._fte_pressure.running_attempts == {str(FteTaskAttemptId.coerce(query_b_attempt))}


def test_fte_existing_fragment_metadata_merge_serializes_with_partition_add(monkeypatch):
    handle = RayWorkerActorHandle(
        _FakeActor(),
        memory_capacity_bytes=1 << 60,
        worker_id="worker-metadata-merge",
    )
    query_id = "query-metadata-merge"
    fragment_id = f"{query_id}:node:7"
    key = (query_id, fragment_id)
    stage = FteFragmentExecution(
        query_id,
        0,
        fragment_id=fragment_id,
        context={"scan_task_nodes": "7,8"},
        task_memory_bytes=64,
    )
    first = stage.add_partition(0)
    merge_iteration_started = threading.Event()
    add_entered = threading.Event()

    class _LockCheckingPartitions(dict):
        def values(self):
            assert stage._state_lock_owned_by_current_thread()
            is_registry_lock_owned = getattr(fragment_submission_mod._FTE_REGISTRY_LOCK, "_is_owned", None)
            assert callable(is_registry_lock_owned) and not is_registry_lock_owned()
            merge_iteration_started.set()
            assert add_entered.wait(2.0)
            return super().values()

    stage.partitions = _LockCheckingPartitions(stage.partitions)
    monkeypatch.setitem(fragment_submission_mod._FTE_FRAGMENT_EXECUTIONS, key, stage)

    def merge_metadata():
        return handle._get_or_create_fte_fragment_execution(
            {
                "query_id": query_id,
                "fragment_id": fragment_id,
                "task_context_info": {"metadata": "merged"},
                "exchange_sink_instance": {"sink": "merged"},
            },
            dynamic_scan_sources={"7"},
            dynamic_exchange_sources={"8"},
        )

    def add_partition():
        add_entered.set()
        return stage.add_partition(1)

    with ThreadPoolExecutor(max_workers=2) as executor:
        merged = executor.submit(merge_metadata)
        assert merge_iteration_started.wait(2.0)
        added = executor.submit(add_partition)
        assert merged.result(timeout=2.0) is stage
        second = added.result(timeout=2.0)

    assert first.descriptor.source_node_ids == {"7", "8"}
    assert second.descriptor.source_node_ids == {"7", "8"}
    assert second.descriptor.task_context_info == {
        "metadata": "merged",
        "exchange_sink_instance": {"sink": "merged"},
    }
    assert second.descriptor.exchange_sink_instance == {"sink": "merged"}


def test_available_fte_workers_snapshots_registry_before_concurrent_removal(monkeypatch):
    iteration_started = threading.Event()
    removal_started = threading.Event()

    class _Worker:
        _fte_healthy = True

        def __init__(self, worker_id):
            self.worker_id = worker_id

    class _LockCheckingRegistry(dict):
        def items(self):
            is_owned = getattr(worker_selection_mod._FTE_REGISTRY_LOCK, "_is_owned", None)
            assert callable(is_owned) and is_owned()
            iteration_started.set()
            assert removal_started.wait(2.0)
            return super().items()

    worker_a = _Worker("worker-a")
    worker_b = _Worker("worker-b")
    registry = _LockCheckingRegistry({"worker-a": worker_a, "worker-b": worker_b})
    monkeypatch.setattr(worker_selection_mod, "_FTE_WORKER_HANDLES", registry)

    def remove_worker():
        removal_started.set()
        with worker_selection_mod._FTE_REGISTRY_LOCK:
            registry.pop("worker-b")

    with ThreadPoolExecutor(max_workers=2) as executor:
        snapshot = executor.submit(
            worker_selection_mod.available_fte_workers,
            worker_a,
            worker_a.worker_id,
        )
        assert iteration_started.wait(2.0)
        removal = executor.submit(remove_worker)
        workers = snapshot.result(timeout=2.0)
        removal.result(timeout=2.0)

    assert workers == [worker_a, worker_b]
    assert registry == {"worker-a": worker_a}


def test_fte_node_wait_placement_rechecks_terminal_partition_after_admission(monkeypatch):
    handle = RayWorkerActorHandle(
        _FakeActor(),
        memory_capacity_bytes=1 << 60,
        worker_id="worker-placement-terminal",
    )
    query_id = "query-placement-terminal"
    fragment_id = f"{query_id}:node:7"
    stage = FteFragmentExecution(
        query_id,
        0,
        fragment_id=fragment_id,
        context={
            "resource_query_id": query_id,
            "resource_stage_id": f"stage:{query_id}:node:7:fte",
        },
        task_memory_bytes=64,
    )
    partition = stage.add_partition(0)
    next_partition = stage.add_partition(1)
    _register_test_query_graph(query_id, [fragment_id])
    monkeypatch.setitem(
        worker_handle_mod._FTE_FRAGMENT_EXECUTIONS,
        (query_id, fragment_id),
        stage,
    )
    admission_started = threading.Event()
    terminal_committed = threading.Event()
    reservation_calls = []

    class _PlacementManager:
        def acquire(self, **kwargs):
            reservation_calls.append(kwargs)
            return object()

    handle._fte_worker_placement_manager = _PlacementManager()

    real_admit = worker_placement_mod._admit_fte_partition_node_wait

    def admit(*args, **kwargs):
        admitted = real_admit(*args, **kwargs)
        assert admitted is True
        admission_started.set()
        assert terminal_committed.wait(2.0)
        return admitted

    monkeypatch.setattr(worker_placement_mod, "_admit_fte_partition_node_wait", admit)

    def finish_partition():
        assert admission_started.wait(2.0)
        with stage._state_lock:
            partition.finished = True
            partition._invalidate_placement()
        terminal_committed.set()

    with ThreadPoolExecutor(max_workers=2) as executor:
        placement = executor.submit(
            handle._try_reserve_fte_partition_for_node_wait,
            query_id,
            fragment_id,
            partition,
            fragment_execution=stage,
        )
        finished = executor.submit(finish_partition)
        placement.result(timeout=2.0)
        finished.result(timeout=2.0)

    assert partition.finished is True
    assert partition.node_wait_started_at is None
    assert reservation_calls == []
    assert fte_fragment_scheduler_mod.fte_submission_window_snapshot()["probes"] == {}
    assert (
        fte_fragment_scheduler_mod.admit_fte_partition_submission(
            query_id,
            fragment_id,
            next_partition.task_id.partition_id,
        )
        is True
    )
    assert (
        fte_fragment_scheduler_mod.release_fte_partition_submission(
            query_id,
            fragment_id,
            next_partition.task_id.partition_id,
        )
        is True
    )


def test_fte_ready_attempt_releases_admission_when_partition_finishes_before_handoff(monkeypatch):
    query_id = "query-ready-attempt-terminal"
    fragment_id = f"{query_id}:node:7"
    stage = FteFragmentExecution(
        query_id,
        0,
        fragment_id=fragment_id,
        context={
            "resource_query_id": query_id,
            "resource_stage_id": f"stage:{query_id}:node:7:fte",
        },
        task_memory_bytes=64,
    )
    partition = stage.add_partition(0)
    next_partition = stage.add_partition(1)
    with stage._state_lock:
        partition.mark_ready_for_execution()
    _register_test_query_graph(query_id, [fragment_id])
    monkeypatch.setitem(
        worker_handle_mod._FTE_FRAGMENT_EXECUTIONS,
        (query_id, fragment_id),
        stage,
    )
    admission_started = threading.Event()
    terminal_committed = threading.Event()
    reservation_calls = []

    def admit(candidate):
        admitted = fte_fragment_scheduler_mod._admit_fte_partition_node_wait(
            query_id,
            candidate,
            stage,
        )
        assert admitted is True
        admission_started.set()
        assert terminal_committed.wait(2.0)
        return admitted

    stage.attempt_admission_callback = admit
    stage.attempt_admission_abandon_callback = lambda candidate: (
        fte_fragment_scheduler_mod.release_fte_partition_submission(
            query_id,
            fragment_id,
            candidate.task_id.partition_id,
        )
    )
    stage.worker_reservation_callback = lambda candidate: reservation_calls.append(candidate) or True

    def finish_partition():
        assert admission_started.wait(2.0)
        with stage._state_lock:
            partition.finished = True
            partition._invalidate_placement()
        terminal_committed.set()

    with ThreadPoolExecutor(max_workers=2) as executor:
        scheduled = executor.submit(stage.schedule_next_pending_partition)
        finished = executor.submit(finish_partition)
        assert scheduled.result(timeout=2.0) is None
        finished.result(timeout=2.0)

    assert reservation_calls == []
    assert fte_fragment_scheduler_mod.fte_submission_window_snapshot()["probes"] == {}
    assert (
        fte_fragment_scheduler_mod.admit_fte_partition_submission(
            query_id,
            fragment_id,
            next_partition.task_id.partition_id,
        )
        is True
    )
    assert (
        fte_fragment_scheduler_mod.release_fte_partition_submission(
            query_id,
            fragment_id,
            next_partition.task_id.partition_id,
        )
        is True
    )


def test_fte_async_reservation_releases_admission_for_terminal_partition(monkeypatch):
    handle = RayWorkerActorHandle(
        _FakeActor(),
        memory_capacity_bytes=1 << 60,
        worker_id="worker-async-reservation-terminal",
    )
    query_id = "query-async-reservation-terminal"
    fragment_id = f"{query_id}:node:7"
    stage = FteFragmentExecution(
        query_id,
        0,
        fragment_id=fragment_id,
        context={
            "resource_query_id": query_id,
            "resource_stage_id": f"stage:{query_id}:node:7:fte",
        },
        task_memory_bytes=64,
    )
    partition = stage.add_partition(0)
    _register_test_query_graph(query_id, [fragment_id])
    monkeypatch.setitem(
        worker_handle_mod._FTE_FRAGMENT_EXECUTIONS,
        (query_id, fragment_id),
        stage,
    )
    assert (
        fte_fragment_scheduler_mod.admit_fte_partition_submission(
            query_id,
            fragment_id,
            partition.task_id.partition_id,
        )
        is True
    )
    with stage._state_lock:
        partition.finished = True
        partition._invalidate_placement()

    assert (
        handle._request_fte_worker_reservation_for_partition(
            query_id,
            fragment_id,
            stage,
            partition,
        )
        is False
    )
    assert fte_fragment_scheduler_mod.fte_submission_window_snapshot()["probes"] == {}


def test_fte_node_wait_placement_releases_reservation_when_partition_finishes_before_commit(monkeypatch):
    handle = RayWorkerActorHandle(
        _FakeActor(),
        memory_capacity_bytes=1 << 60,
        worker_id="worker-placement-rollback",
    )
    query_id = "query-placement-rollback"
    fragment_id = f"{query_id}:node:7"
    stage = FteFragmentExecution(
        query_id,
        0,
        fragment_id=fragment_id,
        task_memory_bytes=64,
    )
    partition = stage.add_partition(0)
    reservation_started = threading.Event()
    terminal_committed = threading.Event()
    releases = []

    class _PlacementManager:
        def acquire(self, **_kwargs):
            reservation_started.set()
            assert terminal_committed.wait(2.0)
            return object()

    handle._fte_worker_placement_manager = _PlacementManager()
    monkeypatch.setattr(worker_placement_mod, "_admit_fte_partition_node_wait", lambda *_args: True)
    monkeypatch.setattr(
        worker_placement_mod.FteWorkerPlacementManager,
        "release_owner",
        lambda **kwargs: releases.append(kwargs),
    )

    def finish_partition():
        assert reservation_started.wait(2.0)
        with stage._state_lock:
            partition.finished = True
            partition._invalidate_placement()
        terminal_committed.set()

    with ThreadPoolExecutor(max_workers=2) as executor:
        placement = executor.submit(
            handle._try_reserve_fte_partition_for_node_wait,
            query_id,
            fragment_id,
            partition,
            fragment_execution=stage,
        )
        finished = executor.submit(finish_partition)
        placement.result(timeout=2.0)
        finished.result(timeout=2.0)

    assert partition.finished is True
    assert len(releases) == 1
    assert releases[0] == {
        "query_id": query_id,
        "fragment_id": fragment_id,
        "partition_id": 0,
        "terminal": False,
    }


def test_fte_async_placement_releases_reservation_when_partition_finishes_before_commit(monkeypatch):
    handle = RayWorkerActorHandle(
        _FakeActor(),
        memory_capacity_bytes=1 << 60,
        worker_id="worker-async-placement-rollback",
    )
    query_id = "query-async-placement-rollback"
    fragment_id = f"{query_id}:node:7"
    stage = FteFragmentExecution(
        query_id,
        0,
        fragment_id=fragment_id,
        context={
            "resource_query_id": query_id,
            "resource_stage_id": f"stage:{query_id}:node:7:fte",
        },
        task_memory_bytes=64,
    )
    partition = stage.add_partition(0)
    monkeypatch.setitem(
        fragment_submission_mod._FTE_FRAGMENT_EXECUTIONS,
        (query_id, fragment_id),
        stage,
    )
    future, created = handle._fte_worker_placement_manager.request_async(
        query_id=query_id,
        fragment_execution_id=stage.fragment_execution_id,
        fragment_id=fragment_id,
        partition_id=0,
        memory_requirement_bytes=64,
        execution_class=partition.execution_class,
    )
    assert created is True
    reservation_started = threading.Event()
    terminal_committed = threading.Event()
    releases = []

    def acquire(**_kwargs):
        reservation_started.set()
        assert terminal_committed.wait(2.0)
        return object()

    monkeypatch.setattr(handle._fte_worker_placement_manager, "acquire", acquire)
    monkeypatch.setattr(
        worker_placement_mod.FteWorkerPlacementManager,
        "release_owner",
        lambda **kwargs: releases.append(kwargs),
    )

    def finish_partition():
        assert reservation_started.wait(2.0)
        with stage._state_lock:
            partition.finished = True
            partition._invalidate_placement()
        terminal_committed.set()

    with ThreadPoolExecutor(max_workers=2) as executor:
        placement = executor.submit(
            handle._try_complete_fte_worker_reservation_future,
            future,
            partition=partition,
        )
        finished = executor.submit(finish_partition)
        assert placement.result(timeout=2.0) is False
        finished.result(timeout=2.0)

    assert future.cancelled() is True
    assert releases == [
        {
            "query_id": query_id,
            "fragment_id": fragment_id,
            "partition_id": 0,
            "terminal": False,
        }
    ]


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


def test_fte_create_promotes_reservation_to_running_atomically(monkeypatch):
    actor = _FakeActor()
    handle = RayWorkerActorHandle(actor, memory_capacity_bytes=15, worker_id="worker-0")
    publish_started = threading.Event()
    allow_publish = threading.Event()
    query_b_started = threading.Event()
    query_b_done = threading.Event()
    errors = []
    handles_by_query = {}
    original_record_started = handle.record_fte_task_started

    def record_started_with_barrier(attempt_id, request):
        attempt = FteTaskAttemptId.coerce(attempt_id)
        if attempt.task_id.query_id == "query-atomic-a":
            publish_started.set()
            if not allow_publish.wait(2.0):
                raise RuntimeError("timed out waiting to publish running pressure")
        return original_record_started(attempt_id, request)

    monkeypatch.setattr(handle, "record_fte_task_started", record_started_with_barrier)

    def submit(query_id, node_id):
        try:
            if query_id == "query-atomic-b":
                query_b_started.set()
            handles_by_query[query_id] = handle.submit_tasks(
                [
                    _FakeTask(
                        name=f"scan-{query_id}",
                        context={"query_id": query_id, "node_id": node_id},
                        inputs={node_id: {"kind": "scan_task", "data": query_id.encode()}},
                    )
                ]
            )
        except BaseException as exc:
            errors.append(exc)
        finally:
            if query_id == "query-atomic-b":
                query_b_done.set()

    thread_a = threading.Thread(target=submit, args=("query-atomic-a", "7"))
    thread_b = threading.Thread(target=submit, args=("query-atomic-b", "8"))
    thread_a.start()
    try:
        assert publish_started.wait(2.0)
        thread_b.start()
        assert query_b_started.wait(2.0)
        query_b_completed_during_handoff = query_b_done.wait(0.1)
    finally:
        allow_publish.set()
        thread_a.join(2.0)
        if thread_b.ident is not None:
            thread_b.join(2.0)

    assert thread_a.is_alive() is False
    assert thread_b.is_alive() is False
    assert errors == []
    assert query_b_completed_during_handoff is False
    assert [str(item.task_id) for item in handles_by_query["query-atomic-a"]] == ["query-atomic-a.0.0.0"]
    assert handles_by_query["query-atomic-b"] == []
    create_requests = _create_requests(actor)
    assert [request["task_id"]["query_id"] for request in create_requests] == ["query-atomic-a"]
    assert handle.fte_pressure_stats()["total_memory_bytes"] == 10


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
    non_matching = RayWorkerActorHandle(
        actor0,
        memory_capacity_bytes=1 << 60,
        worker_id="manager-a:node-a:0",
        host="aaa",
    )
    matching = RayWorkerActorHandle(
        actor1,
        memory_capacity_bytes=1 << 60,
        worker_id="manager-a:node-b:0",
        host="zzz",
    )

    selected = non_matching._select_fte_worker(
        node_requirements=NodeRequirements(host="zzz"),
    )

    assert selected is matching


def test_fte_worker_registry_rejects_duplicate_worker_identity():
    first = RayWorkerActorHandle(
        _FakeActor(),
        memory_capacity_bytes=1 << 60,
        worker_id="manager-a:node-a:0",
        host="10.0.0.1",
    )

    with pytest.raises(RuntimeError, match="FTE worker id is already registered: manager-a:node-a:0"):
        RayWorkerActorHandle(
            _FakeActor(),
            memory_capacity_bytes=1 << 60,
            worker_id="manager-a:node-a:0",
            host="10.0.0.1",
        )

    assert worker_handle_mod._FTE_WORKER_HANDLES["manager-a:node-a:0"] is first
    assert first._fte_healthy is True


def test_fte_worker_selection_stays_with_manager_scope():
    current = RayWorkerActorHandle(
        _FakeActor(),
        memory_capacity_bytes=1 << 60,
        worker_id="manager-a:node-z:0",
        manager_instance_id="manager-a",
    )
    RayWorkerActorHandle(
        _FakeActor(),
        memory_capacity_bytes=1 << 60,
        worker_id="manager-b:node-a:0",
        manager_instance_id="manager-b",
    )

    assert current._select_fte_worker() is current


def test_fte_registry_stats_exposes_worker_topology():
    worker_id = "manager-a:node-a:0"
    RayWorkerActorHandle(
        _FakeActor(),
        memory_capacity_bytes=1 << 60,
        worker_id=worker_id,
        node_id="node-a",
        host="10.0.0.1",
        manager_instance_id="manager-a",
    )

    worker_stats = worker_handle_mod.fte_registry_stats()["workers"][worker_id]

    assert worker_stats["worker_id"] == worker_id
    assert worker_stats["manager_instance_id"] == "manager-a"
    assert worker_stats["node_id"] == "node-a"
    assert worker_stats["host"] == "10.0.0.1"


def test_fte_worker_failure_payload_exposes_worker_topology():
    worker_id = "manager-a:node-a:0"
    handle = RayWorkerActorHandle(
        _FakeActor(),
        memory_capacity_bytes=1 << 60,
        worker_id=worker_id,
        node_id="node-a",
        host="10.0.0.1",
        manager_instance_id="manager-a",
    )

    failure = fte_fragment_scheduler_mod._worker_failure_payload(
        worker_id,
        RuntimeError("worker lost"),
        worker_incarnation_id=handle.worker_incarnation_id,
    )

    assert failure["worker_id"] == worker_id
    assert failure["manager_instance_id"] == "manager-a"
    assert failure["node_id"] == "node-a"
    assert failure["host"] == "10.0.0.1"


def test_fte_worker_command_debug_fields_exposes_worker_topology():
    worker_id = "manager-a:node-a:0"
    handle = RayWorkerActorHandle(
        _FakeActor(),
        memory_capacity_bytes=1 << 60,
        worker_id=worker_id,
        node_id="node-a",
        host="10.0.0.1",
        manager_instance_id="manager-a",
    )
    command = SimpleNamespace(worker=handle, worker_id=worker_id)

    assert worker_commands_mod._fte_worker_command_debug_fields(command) == {
        "worker_id": worker_id,
        "manager_instance_id": "manager-a",
        "node_id": "node-a",
        "host": "10.0.0.1",
    }


def test_fte_worker_failure_replacement_stays_with_manager_scope():
    same_manager = RayWorkerActorHandle(
        _FakeActor(),
        memory_capacity_bytes=1 << 60,
        worker_id="manager-a:node-z:0",
        manager_instance_id="manager-a",
    )
    RayWorkerActorHandle(
        _FakeActor(),
        memory_capacity_bytes=1 << 60,
        worker_id="manager-b:node-a:0",
        manager_instance_id="manager-b",
    )

    replacement = fte_fragment_scheduler_mod._select_replacement_fte_worker(
        "manager-a:failed:0",
        exclude_worker_incarnation_ids={"manager-a:failed:0": "failed-incarnation"},
        manager_instance_id="manager-a",
    )

    assert replacement is same_manager


def test_fte_worker_quarantine_rejects_cross_manager_identity():
    other_manager = RayWorkerActorHandle(
        _FakeActor(),
        memory_capacity_bytes=1 << 60,
        worker_id="manager-b:node-a:0",
        manager_instance_id="manager-b",
    )

    worker_failures_mod.quarantine_fte_worker(
        other_manager.worker_id,
        manager_instance_id="manager-a",
        worker_incarnation_id=other_manager.worker_incarnation_id,
    )

    assert other_manager._fte_healthy is True


def test_fte_worker_failure_deduplicates_each_worker_incarnation():
    scheduler = worker_handle_mod._FTE_SCHEDULERS.get_or_create("query-worker-incarnation-dedup")

    assert scheduler.record_worker_failure("worker-0", worker_incarnation_id="incarnation-0") is True
    assert scheduler.record_worker_failure("worker-0", worker_incarnation_id="incarnation-0") is False
    assert scheduler.record_worker_failure("worker-0", worker_incarnation_id="incarnation-1") is True
    assert scheduler.stats().failed_worker_count == 2


def test_fte_stale_worker_quarantine_does_not_fence_replacement():
    replacement = RayWorkerActorHandle(
        _FakeActor(),
        memory_capacity_bytes=1 << 60,
        worker_id="manager-a:node-a:0",
        manager_instance_id="manager-a",
    )

    worker_failures_mod.quarantine_fte_worker(
        replacement.worker_id,
        manager_instance_id=replacement.manager_instance_id,
        worker_incarnation_id="retired-incarnation",
    )

    assert replacement._fte_healthy is True


def test_fte_blank_manager_scope_remains_legacy_and_fail_closed():
    legacy = RayWorkerActorHandle(
        _FakeActor(),
        memory_capacity_bytes=1 << 60,
        worker_id="legacy#0",
    )
    managed = RayWorkerActorHandle(
        _FakeActor(),
        memory_capacity_bytes=1 << 60,
        worker_id="manager-b:node-a:0",
        manager_instance_id="manager-b",
    )

    worker_failures_mod.quarantine_fte_worker(
        managed.worker_id,
        manager_instance_id="   ",
        worker_incarnation_id=managed.worker_incarnation_id,
    )
    scheduled = fte_fragment_scheduler_mod._mark_fte_worker_failed(
        managed.worker_id,
        fte_fragment_scheduler_mod._worker_failure_payload(
            managed.worker_id,
            RuntimeError("planned failure"),
            worker_incarnation_id=managed.worker_incarnation_id,
        ),
        manager_instance_id="   ",
        worker_incarnation_id=managed.worker_incarnation_id,
    )

    assert scheduled == []
    assert worker_handle_mod._FTE_WORKER_HANDLES[managed.worker_id] is managed
    assert managed._fte_healthy is True
    worker_failures_mod.quarantine_fte_worker(
        legacy.worker_id,
        manager_instance_id="   ",
        worker_incarnation_id=legacy.worker_incarnation_id,
    )
    assert legacy._fte_healthy is False


def test_fte_memory_pressure_skips_worker_without_budget(monkeypatch):
    handle = RayWorkerActorHandle(
        _FakeActor(),
        memory_capacity_bytes=1 << 60,
        worker_id="legacy#0",
    )
    monkeypatch.setattr(
        worker_transitions_mod,
        "_fte_effective_worker_memory_budget_bytes",
        lambda _worker, _execution_class: None,
    )

    assert handle._revoke_fte_speculative_tasks_for_memory_pressure_direct() == []


def test_fte_query_scheduler_rejects_cross_manager_rebinding():
    first = RayWorkerActorHandle(
        _FakeActor(),
        memory_capacity_bytes=1 << 60,
        worker_id="manager-a:node-a:0",
        manager_instance_id="manager-a",
    )
    second = RayWorkerActorHandle(
        _FakeActor(),
        memory_capacity_bytes=1 << 60,
        worker_id="manager-b:node-a:0",
        manager_instance_id="manager-b",
    )
    scheduler = worker_handle_mod._FTE_SCHEDULERS.get_or_create("query-manager-scope")
    first._bind_fte_scheduler_handlers(scheduler)

    with pytest.raises(RuntimeError, match="FTE query scheduler manager ownership mismatch"):
        second._bind_fte_scheduler_handlers(scheduler)


def test_fte_global_worker_failure_does_not_claim_unbound_scheduler():
    first = RayWorkerActorHandle(
        _FakeActor(),
        memory_capacity_bytes=1 << 60,
        worker_id="manager-a:node-a:0",
        manager_instance_id="manager-a",
    )
    second = RayWorkerActorHandle(
        _FakeActor(),
        memory_capacity_bytes=1 << 60,
        worker_id="manager-b:node-a:0",
        manager_instance_id="manager-b",
    )
    scheduler = worker_handle_mod._FTE_SCHEDULERS.get_or_create("query-unbound-manager-scope")

    first.mark_fte_worker_failed(
        first.worker_id,
        RuntimeError("planned failure"),
        worker_incarnation_id=first.worker_incarnation_id,
    )
    second._bind_fte_scheduler_handlers(scheduler)

    assert scheduler.is_owned_by_manager_instance("manager-b")


@pytest.mark.parametrize("failed_manager_instance_id", ["manager-a", ""])
def test_fte_worker_failure_event_rejects_cross_manager_scheduler(failed_manager_instance_id):
    failed = RayWorkerActorHandle(
        _FakeActor(),
        memory_capacity_bytes=1 << 60,
        worker_id=f"{failed_manager_instance_id or 'legacy'}:node-a:0",
        manager_instance_id=failed_manager_instance_id,
    )
    owner = RayWorkerActorHandle(
        _FakeActor(),
        memory_capacity_bytes=1 << 60,
        worker_id="manager-b:node-a:0",
        manager_instance_id="manager-b",
    )
    scheduler = worker_handle_mod._FTE_SCHEDULERS.get_or_create("query-cross-manager-failure")
    owner._bind_fte_scheduler_handlers(scheduler)

    scheduled = worker_failures_mod.mark_fte_worker_failed_for_event(
        WorkerFailed(
            query_id=scheduler.query_id,
            worker_id=failed.worker_id,
            worker_incarnation_id=failed.worker_incarnation_id,
            manager_instance_id=failed_manager_instance_id,
            error=RuntimeError("planned failure"),
        )
    )

    assert scheduled == []
    assert scheduler.stats().failed_worker_count == 0
    assert worker_handle_mod._FTE_WORKER_HANDLES[failed.worker_id] is failed
    assert failed._fte_healthy is True


def test_stale_worker_shutdown_does_not_fail_current_registry_owner(monkeypatch):
    stale = RayWorkerActorHandle(
        _FakeActor(),
        memory_capacity_bytes=1 << 60,
        worker_id="manager-a:node-a:0",
        host="10.0.0.1",
    )
    current = SimpleNamespace(_fte_healthy=True)
    failure_calls = []
    monkeypatch.setattr(
        stale,
        "mark_fte_worker_failed",
        lambda worker_id, error, *, worker_incarnation_id: failure_calls.append(
            (worker_id, worker_incarnation_id, error)
        ),
    )
    with worker_handle_mod._FTE_REGISTRY_LOCK:
        worker_handle_mod._FTE_WORKER_HANDLES[stale.worker_id] = current

    stale._begin_worker_shutdown()

    assert failure_calls == []
    assert worker_handle_mod._FTE_WORKER_HANDLES[stale.worker_id] is current
    assert current._fte_healthy is True


def test_worker_failure_does_not_retire_replacement_registered_during_quiescence(monkeypatch):
    worker_id = "manager-a:node-a:0"
    actor = _FakeActor()
    failed = RayWorkerActorHandle(
        actor,
        memory_capacity_bytes=1 << 60,
        worker_id=worker_id,
        manager_instance_id="manager-a",
    )
    retirement_calls = []
    replacement = SimpleNamespace(
        _fte_healthy=True,
        _retire_from_manager_for_failure=lambda: retirement_calls.append(True) or True,
        actor_handle=object(),
        manager_instance_id="manager-a",
        worker_incarnation_id="replacement-incarnation",
    )

    class _ReplaceRegistryOwnerOnPrepare:
        def remote(self):
            with worker_handle_mod._FTE_REGISTRY_LOCK:
                assert worker_handle_mod._FTE_WORKER_HANDLES[worker_id] is failed
                worker_handle_mod._FTE_WORKER_HANDLES[worker_id] = replacement
            future = Future()
            future.set_result(None)
            return SimpleNamespace(future=lambda: future)

    actor.prepare_shutdown = _ReplaceRegistryOwnerOnPrepare()
    kill_calls = []
    monkeypatch.setattr(worker_handle_mod.ray, "kill", lambda target: kill_calls.append(target))

    scheduled = fte_fragment_scheduler_mod._mark_fte_worker_failed(
        worker_id,
        {"error_code": "WORKER_LOST", "message": "planned concurrent replacement"},
        manager_instance_id="manager-a",
        worker_incarnation_id=failed.worker_incarnation_id,
    )

    assert scheduled == []
    assert worker_handle_mod._FTE_WORKER_HANDLES[worker_id] is replacement
    assert replacement._fte_healthy is True
    assert retirement_calls == []
    assert kill_calls == []


def test_worker_failure_kills_retired_actor_when_retry_bookkeeping_raises(monkeypatch):
    actor = _FakeActor()
    failed = RayWorkerActorHandle(
        actor,
        memory_capacity_bytes=1 << 60,
        worker_id="manager-a:node-a:0",
        manager_instance_id="manager-a",
    )
    failed._retire_from_manager_for_failure = lambda: True
    monkeypatch.setattr(
        fte_fragment_scheduler_mod,
        "_running_partition_requirements_on_fte_workers",
        lambda *_args, **_kwargs: {
            ("query-retry-error", "fragment-retry-error", 0): (
                None,
                fte_fragment_scheduler_mod.FteTaskExecutionClass.STANDARD,
                None,
            )
        },
    )

    def fail_replacement_selection(*_args, **_kwargs):
        raise RuntimeError("planned replacement selection failure")

    monkeypatch.setattr(
        fte_fragment_scheduler_mod,
        "_select_replacement_fte_worker",
        fail_replacement_selection,
    )
    kill_calls = []
    monkeypatch.setattr(worker_handle_mod.ray, "kill", lambda target: kill_calls.append(target))

    with pytest.raises(RuntimeError, match="planned replacement selection failure"):
        fte_fragment_scheduler_mod._mark_fte_worker_failed(
            failed.worker_id,
            {"error_code": "WORKER_LOST", "message": "planned failure"},
            manager_instance_id="manager-a",
            worker_incarnation_id=failed.worker_incarnation_id,
        )

    assert failed.worker_id not in worker_handle_mod._FTE_WORKER_HANDLES
    assert kill_calls == [actor]


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
    monkeypatch.setattr(fte_execution_mod.time, "time", lambda: 42.0)
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


class _BlockingStringError(RuntimeError):
    def __init__(self, message, entered, release):
        super().__init__(message)
        self._entered = entered
        self._release = release

    def __str__(self):
        self._entered.set()
        assert self._release.wait(5.0)
        return super().__str__()


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


def test_fte_worker_reservation_completion_racing_query_drop_is_ignored(monkeypatch):
    from duckdb.runners.ray import fragment_worker_events as worker_events_mod

    query_id = "query-reservation-during-drop"
    actor, handle, pending_key, pending_future = _submit_strict_worker_reservation_pending_pair(
        monkeypatch,
        query_id,
    )
    pending_future.set_result(_completed_test_reservation(pending_future, handle))
    scheduler = worker_handle_mod._FTE_SCHEDULERS.get(query_id)
    assert scheduler is not None

    reservation_removed = threading.Event()
    allow_completion = threading.Event()
    original_remove = worker_events_mod.remove_pending_fte_worker_reservation_if_current

    def blocked_remove(key, future):
        removed = original_remove(key, future)
        if removed:
            reservation_removed.set()
            assert allow_completion.wait(5.0)
        return removed

    monkeypatch.setattr(
        worker_events_mod,
        "remove_pending_fte_worker_reservation_if_current",
        blocked_remove,
    )
    with ThreadPoolExecutor(max_workers=1) as executor:
        completion = executor.submit(scheduler.drain)
        assert reservation_removed.wait(5.0)
        try:
            handle.fte_drop_query(query_id)
        finally:
            allow_completion.set()
        assert completion.result(timeout=5.0) == []

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


def test_fte_worker_reservation_completion_is_atomic_with_pending_drain(monkeypatch):
    from duckdb.runners.ray import fragment_worker_events as worker_events_mod

    query_id = "query-reservation-completion-race"
    actor, handle, pending_key, pending_future = _submit_strict_worker_reservation_pending_pair(
        monkeypatch,
        query_id,
    )
    running = handle.pop_fte_result_handles(query_id)
    assert len(running) == 1

    first_reservation_removed = threading.Event()
    allow_first_attempt_start = threading.Event()
    pending_drain_started = threading.Event()
    command_handoff_lock_owned = []
    remove_generations = []
    original_remove = worker_events_mod.remove_pending_fte_worker_reservation_if_current

    def blocked_remove(key, future):
        removed = original_remove(key, future)
        if removed:
            remove_generations.append(future.reservation_generation)
        if removed and len(remove_generations) == 1:
            first_reservation_removed.set()
            assert allow_first_attempt_start.wait(5.0)
        return removed

    monkeypatch.setattr(
        worker_events_mod,
        "remove_pending_fte_worker_reservation_if_current",
        blocked_remove,
    )
    with ThreadPoolExecutor(max_workers=2) as executor:
        completion = executor.submit(handle.record_fte_task_terminal, running[0].task_id)
        assert first_reservation_removed.wait(5.0)

        fragment_execution = worker_handle_mod._FTE_FRAGMENT_EXECUTIONS[(query_id, pending_key[1])]
        original_pop_worker_commands = fragment_execution.pop_worker_commands

        def observed_pop_worker_commands():
            command_handoff_lock_owned.append(fragment_execution._attempt_scheduling_lock._is_owned())
            return original_pop_worker_commands()

        monkeypatch.setattr(fragment_execution, "pop_worker_commands", observed_pop_worker_commands)

        def drain_pending():
            pending_drain_started.set()
            return worker_handle_mod.request_fte_pending_task_drain()

        drain = executor.submit(drain_pending)
        assert pending_drain_started.wait(5.0)
        allow_first_attempt_start.set()
        completion.result(timeout=5.0)
        drain.result(timeout=5.0)

    assert remove_generations == [pending_future.reservation_generation]
    assert fragment_execution.partitions[1].running_attempt is not None
    assert worker_handle_mod._FTE_PARTITION_OWNERS[pending_key] is handle
    assert pending_key in worker_handle_mod._FTE_PARTITION_TASK_LEASES
    assert pending_key not in worker_handle_mod._FTE_PENDING_WORKER_RESERVATIONS
    assert command_handoff_lock_owned == [True]
    assert [call[1]["task_id"]["partition_id"] for call in actor.fte_calls if call[0] == "create"] == [0, 1]


@pytest.mark.parametrize("failure_point", ["bind", "enqueue"])
def test_fte_worker_reservation_callback_failure_fails_query_and_releases_resources(
    monkeypatch,
    failure_point,
):
    query_id = f"query-reservation-callback-{failure_point}"
    actor, handle, pending_key, pending_future = _submit_strict_worker_reservation_pending_pair(
        monkeypatch,
        query_id,
    )
    running = handle.pop_fte_result_handles(query_id)
    assert len(running) == 1
    scheduler = worker_handle_mod._FTE_SCHEDULERS.get(query_id)
    assert scheduler is not None
    failure = RuntimeError(f"planned reservation callback {failure_point} failure")

    if failure_point == "bind":

        def fail_completion_bind(_scheduler):
            raise failure

        monkeypatch.setattr(handle, "_bind_fte_scheduler_handlers", fail_completion_bind)
    else:
        original_enqueue = scheduler.enqueue

        def fail_completion_enqueue(event, *, priority=False):
            if isinstance(event, WorkerReservationCompleted):
                raise failure
            return original_enqueue(event, priority=priority)

        monkeypatch.setattr(scheduler, "enqueue", fail_completion_enqueue)

    handle.record_fte_task_terminal(running[0].task_id)

    fragment_execution = worker_handle_mod._FTE_FRAGMENT_EXECUTIONS[(query_id, pending_key[1])]
    stats = scheduler.stats()
    assert pending_future.done() is True
    assert pending_key not in worker_handle_mod._FTE_PENDING_WORKER_RESERVATIONS
    assert pending_key not in worker_handle_mod._FTE_PARTITION_OWNERS
    assert pending_key not in worker_handle_mod._FTE_PARTITION_TASK_LEASES
    assert fragment_execution.partitions[pending_key[2]].running_attempt is None
    assert stats.state == "FAILED"
    assert stats.queued_events == 0
    assert stats.failure_reason == (
        f"FTE worker reservation failed: FTE worker reservation completion callback failed: {failure}"
    )
    assert [call[1]["task_id"]["partition_id"] for call in actor.fte_calls if call[0] == "create"] == [0]


def test_fte_worker_reservation_callback_failure_ignores_replacement_generation(monkeypatch):
    query_id = "query-reservation-callback-stale"
    actor, handle, pending_key, old_future = _submit_strict_worker_reservation_pending_pair(
        monkeypatch,
        query_id,
    )
    callback_entered = threading.Event()
    release_callback = threading.Event()

    def fail_completion_bind(_scheduler):
        callback_entered.set()
        assert release_callback.wait(5.0)
        raise RuntimeError("planned stale reservation callback failure")

    monkeypatch.setattr(handle, "_bind_fte_scheduler_handlers", fail_completion_bind)
    with ThreadPoolExecutor(max_workers=1) as executor:
        completion = executor.submit(
            old_future.set_result,
            _completed_test_reservation(old_future, handle),
        )
        assert callback_entered.wait(5.0)
        try:
            with worker_handle_mod._FTE_REGISTRY_LOCK:
                removed_future = worker_handle_mod._FTE_PENDING_WORKER_RESERVATIONS.pop(pending_key)
                fragment_execution = worker_handle_mod._FTE_FRAGMENT_EXECUTIONS[(query_id, pending_key[1])]
            partition = fragment_execution.partitions[pending_key[2]]
            new_future, created = handle._fte_worker_placement_manager.request_async(
                query_id=query_id,
                fragment_execution_id=fragment_execution.fragment_execution_id,
                fragment_id=pending_key[1],
                partition_id=pending_key[2],
                memory_requirement_bytes=partition.memory_requirement_bytes,
                execution_class=partition.execution_class,
                node_requirements=partition.node_requirements,
                node_requirements_wait_started_at=partition.node_wait_started_at,
                on_done=handle._enqueue_fte_worker_reservation_completion,
                on_done_error=handle._handle_fte_worker_reservation_callback_error,
            )
        finally:
            release_callback.set()
        assert completion.result(timeout=5.0) is True

    scheduler = worker_handle_mod._FTE_SCHEDULERS.get(query_id)
    assert scheduler is not None
    assert removed_future is old_future
    assert created is True
    assert new_future.reservation_generation == old_future.reservation_generation + 1
    assert worker_handle_mod._FTE_PENDING_WORKER_RESERVATIONS[pending_key] is new_future
    assert scheduler.stats().state == "RUNNING"
    assert scheduler.stats().queued_events == 0
    assert [call[1]["task_id"]["partition_id"] for call in actor.fte_calls if call[0] == "create"] == [0]


def test_fte_failed_worker_reservation_blocks_concurrent_retry(monkeypatch):
    query_id = "query-reservation-failure-race"
    actor, handle, pending_key, pending_future = _submit_strict_worker_reservation_pending_pair(
        monkeypatch,
        query_id,
    )
    error_format_entered = threading.Event()
    release_error_format = threading.Event()
    pending_future.set_exception(
        _BlockingStringError(
            "planned worker reservation failure",
            error_format_entered,
            release_error_format,
        )
    )
    scheduler = worker_handle_mod._FTE_SCHEDULERS.get(query_id)
    assert scheduler is not None
    completion_errors = []

    def drain_completion():
        try:
            scheduler.drain()
        except BaseException as exc:  # pragma: no cover - asserted below
            completion_errors.append(exc)

    completion_thread = threading.Thread(target=drain_completion)
    completion_thread.start()
    assert error_format_entered.wait(1.0)
    try:
        concurrent_handles = worker_handle_mod.request_fte_pending_task_drain()
        with worker_handle_mod._FTE_REGISTRY_LOCK:
            current_future = worker_handle_mod._FTE_PENDING_WORKER_RESERVATIONS.get(pending_key)
            current_generation = worker_handle_mod._FTE_WORKER_RESERVATION_GENERATIONS[pending_key]
    finally:
        release_error_format.set()
        completion_thread.join(2.0)

    assert completion_thread.is_alive() is False
    assert len(completion_errors) == 1
    assert isinstance(completion_errors[0], RuntimeError)
    assert str(completion_errors[0]) == "FTE worker reservation failed: planned worker reservation failure"
    assert concurrent_handles == []
    assert current_future is pending_future
    assert current_generation == pending_future.reservation_generation
    assert pending_future.done() is True
    assert pending_key not in worker_handle_mod._FTE_PENDING_WORKER_RESERVATIONS
    assert pending_key not in worker_handle_mod._FTE_PARTITION_OWNERS
    assert pending_key not in worker_handle_mod._FTE_PARTITION_TASK_LEASES
    assert scheduler.stats().state == "FAILED"
    assert [call[1]["task_id"]["partition_id"] for call in actor.fte_calls if call[0] == "create"] == [0]


def test_fte_failed_worker_reservation_ignores_replacement_generation(monkeypatch):
    query_id = "query-reservation-failure-stale"
    actor, handle, pending_key, old_future = _submit_strict_worker_reservation_pending_pair(
        monkeypatch,
        query_id,
    )
    error_format_entered = threading.Event()
    release_error_format = threading.Event()
    old_future.set_exception(
        _BlockingStringError(
            "planned stale worker reservation failure",
            error_format_entered,
            release_error_format,
        )
    )
    scheduler = worker_handle_mod._FTE_SCHEDULERS.get(query_id)
    assert scheduler is not None
    completion_results = []
    completion_errors = []

    def drain_completion():
        try:
            completion_results.append(scheduler.drain())
        except BaseException as exc:  # pragma: no cover - asserted below
            completion_errors.append(exc)

    completion_thread = threading.Thread(target=drain_completion)
    completion_thread.start()
    assert error_format_entered.wait(1.0)
    try:
        with worker_handle_mod._FTE_REGISTRY_LOCK:
            removed_future = worker_handle_mod._FTE_PENDING_WORKER_RESERVATIONS.pop(pending_key, None)
            fragment_execution = worker_handle_mod._FTE_FRAGMENT_EXECUTIONS[(query_id, pending_key[1])]
        partition = fragment_execution.partitions[pending_key[2]]
        new_future, created = handle._fte_worker_placement_manager.request_async(
            query_id=query_id,
            fragment_execution_id=fragment_execution.fragment_execution_id,
            fragment_id=pending_key[1],
            partition_id=pending_key[2],
            memory_requirement_bytes=partition.memory_requirement_bytes,
            execution_class=partition.execution_class,
            node_requirements=partition.node_requirements,
            node_requirements_wait_started_at=partition.node_wait_started_at,
            on_done=handle._enqueue_fte_worker_reservation_completion,
        )
    finally:
        release_error_format.set()
        completion_thread.join(2.0)

    assert completion_thread.is_alive() is False
    assert completion_errors == []
    assert completion_results == [[]]
    assert removed_future is old_future
    assert created is True
    assert new_future.reservation_generation == old_future.reservation_generation + 1
    assert worker_handle_mod._FTE_PENDING_WORKER_RESERVATIONS[pending_key] is new_future
    assert [call[1]["task_id"]["partition_id"] for call in actor.fte_calls if call[0] == "create"] == [0]


def test_fte_failed_worker_reservation_racing_query_drop_is_ignored(monkeypatch):
    query_id = "query-reservation-failure-drop"
    actor, handle, pending_key, pending_future = _submit_strict_worker_reservation_pending_pair(
        monkeypatch,
        query_id,
    )
    error_format_entered = threading.Event()
    release_error_format = threading.Event()
    pending_future.set_exception(
        _BlockingStringError(
            "planned worker reservation failure",
            error_format_entered,
            release_error_format,
        )
    )
    scheduler = worker_handle_mod._FTE_SCHEDULERS.get(query_id)
    assert scheduler is not None
    completion_results = []
    completion_errors = []

    def drain_completion():
        try:
            completion_results.append(scheduler.drain())
        except BaseException as exc:  # pragma: no cover - asserted below
            completion_errors.append(exc)

    completion_thread = threading.Thread(target=drain_completion)
    completion_thread.start()
    assert error_format_entered.wait(1.0)
    try:
        drop_result = handle.fte_drop_query(query_id)
    finally:
        release_error_format.set()
        completion_thread.join(2.0)

    assert completion_thread.is_alive() is False
    assert completion_errors == []
    assert completion_results == [[]]
    assert drop_result == {"tasks_removed": 1, "tasks_canceled": 0, "fragments_removed": 2}
    assert pending_key not in worker_handle_mod._FTE_PENDING_WORKER_RESERVATIONS
    assert (query_id, pending_key[1]) not in worker_handle_mod._FTE_FRAGMENT_EXECUTIONS
    assert [call[1]["task_id"]["partition_id"] for call in actor.fte_calls if call[0] == "create"] == [0]


def test_fte_worker_reservation_handle_publication_racing_query_drop_is_ignored(monkeypatch):
    query_id = "query-reservation-handle-drop"
    actor, handle, pending_key, _ = _submit_strict_worker_reservation_pending_pair(
        monkeypatch,
        query_id,
    )
    running = handle.pop_fte_result_handles(query_id)
    assert len(running) == 1
    handle_construction_entered = threading.Event()
    release_handle_construction = threading.Event()

    class _BlockingFteTaskHandle(_FakeFteTaskHandle):
        def __init__(self, task_id, worker_handle):
            handle_construction_entered.set()
            assert release_handle_construction.wait(5.0)
            super().__init__(task_id, worker_handle)

    monkeypatch.setattr(handle, "_fte_task_handle_cls", lambda: _BlockingFteTaskHandle)
    completion_errors = []

    def release_running_capacity():
        try:
            handle.record_fte_task_terminal(running[0].task_id)
        except BaseException as exc:  # pragma: no cover - asserted below
            completion_errors.append(exc)

    completion_thread = threading.Thread(target=release_running_capacity)
    completion_thread.start()
    assert handle_construction_entered.wait(1.0)
    assert [call[1]["task_id"]["partition_id"] for call in actor.fte_calls if call[0] == "create"] == [0, 1]
    try:
        drop_result = handle.fte_drop_query(query_id)
    finally:
        release_handle_construction.set()
        completion_thread.join(2.0)

    assert completion_thread.is_alive() is False
    assert completion_errors == []
    assert drop_result == {"tasks_removed": 1, "tasks_canceled": 0, "fragments_removed": 2}
    assert pending_key not in worker_handle_mod._FTE_PENDING_WORKER_RESERVATIONS
    assert (query_id, pending_key[1]) not in worker_handle_mod._FTE_FRAGMENT_EXECUTIONS
    assert handle.pop_fte_result_handles(query_id) == []


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


@pytest.mark.parametrize(
    ("terminal_state", "terminal_extra", "next_query_a_attempt"),
    [
        ("FINISHED", {}, "query-a.0.1.0"),
        (
            "FAILED",
            {
                "failure": {
                    "error_code": "GENERIC_INTERNAL_ERROR",
                    "message": "retry me",
                }
            },
            "query-a.0.0.1",
        ),
    ],
)
def test_fte_pending_drain_is_fair_across_queries(
    monkeypatch,
    terminal_state,
    terminal_extra,
    next_query_a_attempt,
):
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
                context={
                    "query_id": "query-a",
                    "node_id": "8",
                    "task_execution_class": "STANDARD",
                },
                inputs={"3": {"kind": "exchange_source_task", "data": b"p0"}},
                plan={"plan": "exchange-template"},
            ),
            _FakeTask(
                name="query-a-task-1",
                context={
                    "query_id": "query-a",
                    "node_id": "8",
                    "task_execution_class": "STANDARD",
                },
                inputs={"3": {"kind": "exchange_source_task", "data": b"p1"}},
                plan={"plan": "exchange-template"},
            ),
        ]
    )
    query_b_handles = handle.submit_tasks(
        [
            _FakeTask(
                name="query-b-task-0",
                context={
                    "query_id": "query-b",
                    "node_id": "8",
                    "task_execution_class": "STANDARD",
                },
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

    completion_handles = handle.handle_fte_task_status(
        {
            "state": terminal_state,
            "task_id": query_a_handles[0].task_id.to_dict(),
            "version": 1,
            **terminal_extra,
        }
    )
    first_scheduled = handle.pop_fte_result_handles("query-b")

    assert [str(task_handle.task_id) for task_handle in completion_handles] == ["query-b.0.0.0"]
    assert [str(task_handle.task_id) for task_handle in first_scheduled] == ["query-b.0.0.0"]
    handle.record_fte_task_terminal(first_scheduled[0].task_id)
    second_scheduled = handle.pop_fte_result_handles("query-a")
    assert [str(task_handle.task_id) for task_handle in second_scheduled] == [next_query_a_attempt]


def test_fte_status_refresh_drains_released_capacity_across_queries(monkeypatch):
    monkeypatch.setattr(
        fragment_submission_mod,
        "_split_exchange_source_task_by_partition",
        lambda value: ([(int(value[-1:]), value)], 2, 2, False),
    )
    actor = _FakeActor()
    handle = RayWorkerActorHandle(actor, memory_capacity_bytes=15, worker_id="worker-0")

    def task(query_id):
        return _FakeTask(
            name=f"{query_id}-task-0",
            context={"query_id": query_id, "node_id": "8"},
            inputs={"3": {"kind": "exchange_source_task", "data": b"p0"}},
            plan={"plan": "exchange-template"},
        )

    handle.submit_tasks([task("query-a")])
    assert handle.submit_tasks([task("query-b")]) == []
    assert [str(task_handle.task_id) for task_handle in handle.pop_fte_result_handles("query-a")] == ["query-a.0.0.0"]

    monkeypatch.setattr(
        handle,
        "fte_get_task_status",
        lambda task_id, timeout_s=None: {
            "state": "FINISHED",
            "task_id": task_id,
            "version": 1,
        },
    )
    worker_handle_mod.refresh_fte_running_task_stats("query-a")

    query_a_stage = worker_handle_mod._FTE_FRAGMENT_EXECUTIONS[("query-a", "query-a:node:8")]
    assert query_a_stage.partitions[0].finished is True
    assert [str(task_handle.task_id) for task_handle in handle.pop_fte_result_handles("query-b")] == ["query-b.0.0.0"]


def test_fte_capacity_pump_isolates_failed_query_reservation(monkeypatch):
    monkeypatch.setattr(
        fragment_submission_mod,
        "_split_exchange_source_task_by_partition",
        lambda value: ([(int(value[-1:]), value)], 2, 2, False),
    )
    actor = _FakeActor()
    handle = RayWorkerActorHandle(actor, memory_capacity_bytes=15, worker_id="worker-0")

    def task(query_id, execution_class="STANDARD"):
        return _FakeTask(
            name=f"{query_id}-task-0",
            context={
                "query_id": query_id,
                "node_id": "8",
                "task_execution_class": execution_class,
            },
            inputs={"3": {"kind": "exchange_source_task", "data": b"p0"}},
            plan={"plan": "exchange-template"},
        )

    query_a_handle = handle.submit_tasks([task("query-pump-a")])[0]
    assert handle.submit_tasks([task("query-pump-b")]) == []
    assert handle.submit_tasks([task("query-pump-c", "SPECULATIVE")]) == []
    pending_key = ("query-pump-b", "query-pump-b:node:8", 0)
    pending_future = worker_handle_mod._FTE_PENDING_WORKER_RESERVATIONS[pending_key]
    pending_future.set_exception(RuntimeError("planned query-b reservation failure"))

    handle.record_fte_task_terminal(query_a_handle.task_id)

    query_a_scheduler = worker_handle_mod._FTE_SCHEDULERS.get("query-pump-a")
    query_b_scheduler = worker_handle_mod._FTE_SCHEDULERS.get("query-pump-b")
    assert query_a_scheduler is not None
    assert query_b_scheduler is not None
    assert query_a_scheduler.stats().state == "RUNNING"
    assert query_b_scheduler.stats().state == "FAILED"
    assert "planned query-b reservation failure" in str(query_b_scheduler.stats().failure_reason)
    assert pending_key not in worker_handle_mod._FTE_PENDING_WORKER_RESERVATIONS
    assert [str(task_handle.task_id) for task_handle in handle.pop_fte_result_handles("query-pump-c")] == [
        "query-pump-c.0.0.0"
    ]


def test_fte_speculative_admission_isolates_other_query_scan_failure():
    failed_scheduler = worker_handle_mod._FTE_SCHEDULERS.get_or_create("query-scan-failure")
    healthy_scheduler = worker_handle_mod._FTE_SCHEDULERS.get_or_create("query-scan-healthy")

    class _BrokenFragmentExecution:
        @staticmethod
        def _state_lock_owned_by_current_thread():
            return False

        @staticmethod
        def has_pending_partitions(*_args):
            raise RuntimeError("planned admission scan failure")

    class _SpeculativePartition:
        node_wait_started_at = None
        execution_class = "SPECULATIVE"

    worker_handle_mod._FTE_FRAGMENT_EXECUTIONS[("query-scan-failure", "fragment-0")] = _BrokenFragmentExecution()

    assert (
        worker_handle_mod._admit_fte_partition_node_wait(
            "query-scan-healthy",
            _SpeculativePartition(),
        )
        is False
    )
    assert failed_scheduler.stats().state == "FAILED"
    assert "planned admission scan failure" in str(failed_scheduler.stats().failure_reason)
    assert healthy_scheduler.stats().state == "RUNNING"


def test_fte_resource_waiter_scan_isolates_failed_query():
    resource_query_id = "query-resource-scan"
    failed_query_id = "query-resource-scan-failure"
    healthy_query_id = "query-resource-scan-healthy"
    failed_scheduler = worker_handle_mod._FTE_SCHEDULERS.get_or_create(failed_query_id)
    worker_handle_mod._FTE_SCHEDULERS.get_or_create(healthy_query_id)

    class _FragmentExecution:
        context = {
            "resource_query_id": resource_query_id,
            "resource_stage_id": f"stage:{resource_query_id}:node:8:fte",
        }

        def __init__(self, error=None):
            self.error = error

        def has_pending_partitions(self):
            if self.error is not None:
                raise self.error
            return True

    worker_handle_mod._FTE_FRAGMENT_EXECUTIONS[(failed_query_id, "fragment-0")] = _FragmentExecution(
        RuntimeError("planned resource waiter scan failure")
    )
    worker_handle_mod._FTE_FRAGMENT_EXECUTIONS[(healthy_query_id, "fragment-0")] = _FragmentExecution()

    assert worker_handle_mod._fte_execution_queries_waiting_for_resource(resource_query_id) == (healthy_query_id,)
    assert failed_scheduler.stats().state == "FAILED"
    assert "planned resource waiter scan failure" in str(failed_scheduler.stats().failure_reason)


def test_fte_query_drop_durably_wakes_other_pending_queries(monkeypatch):
    monkeypatch.setattr(
        fragment_submission_mod,
        "_split_exchange_source_task_by_partition",
        lambda value: ([(int(value[-1:]), value)], 2, 2, False),
    )
    actor = _FakeActor()
    handle = RayWorkerActorHandle(actor, memory_capacity_bytes=15, worker_id="worker-0")

    def task(query_id):
        return _FakeTask(
            name=f"{query_id}-task-0",
            context={"query_id": query_id, "node_id": "8"},
            inputs={"3": {"kind": "exchange_source_task", "data": b"p0"}},
            plan={"plan": "exchange-template"},
        )

    assert len(handle.submit_tasks([task("query-drop-a")])) == 1
    assert handle.submit_tasks([task("query-drop-b")]) == []

    handle.fte_drop_query("query-drop-a")

    scheduled = handle.pop_fte_result_handles("query-drop-b")
    assert [str(task_handle.task_id) for task_handle in scheduled] == ["query-drop-b.0.0.0"]


def test_fte_capacity_pump_preserves_create_before_followup_worker_commands(monkeypatch):
    monkeypatch.setattr(
        fragment_submission_mod,
        "_split_exchange_source_task_by_partition",
        lambda value: ([(int(value[-1:]), value)], 2, 2, False),
    )
    create_handoff_entered = threading.Event()
    release_create_handoff = threading.Event()
    actor = _FakeActor()
    handle = RayWorkerActorHandle(actor, memory_capacity_bytes=15, worker_id="worker-0")
    original_execute_commands = handle._execute_fte_fragment_execution_worker_commands

    def block_before_create(fragment_execution, worker_commands):
        if fragment_execution.query_id == "query-command-order" and any(
            command.command_type == "FteCreateTaskCommand" for command in worker_commands
        ):
            create_handoff_entered.set()
            assert release_create_handoff.wait(5.0)
        return original_execute_commands(fragment_execution, worker_commands)

    monkeypatch.setattr(
        handle,
        "_execute_fte_fragment_execution_worker_commands",
        block_before_create,
    )

    def task(query_id):
        return _FakeTask(
            name=f"{query_id}-task-0",
            context={"query_id": query_id, "node_id": "8"},
            inputs={"3": {"kind": "exchange_source_task", "data": b"p0"}},
            plan={"plan": "exchange-template"},
        )

    blocker = handle.submit_tasks([task("query-command-blocker")])[0]
    assert handle.submit_tasks([task("query-command-order")]) == []
    release_errors = []

    def release_capacity():
        try:
            handle.record_fte_task_terminal(blocker.task_id)
        except BaseException as exc:  # pragma: no cover - asserted below
            release_errors.append(exc)

    release_thread = threading.Thread(target=release_capacity)
    release_thread.start()
    assert create_handoff_entered.wait(1.0)

    handle.task_input_stream_exhausted_for_query("query-command-order", ["3"])
    release_create_handoff.set()
    release_thread.join(2.0)

    assert release_thread.is_alive() is False
    assert release_errors == []
    ordered_calls = [
        call[0]
        for call in actor.fte_calls
        if (call[0] == "create" and call[1]["task_id"]["query_id"] == "query-command-order")
        or (call[0] == "no_more_splits" and call[1]["query_id"] == "query-command-order")
    ]
    assert ordered_calls == ["create", "no_more_splits"]


def test_fte_handle_publication_holds_query_lifecycle_through_watcher_registration(monkeypatch):
    monkeypatch.setattr(
        fragment_submission_mod,
        "_split_exchange_source_task_by_partition",
        lambda value: ([(int(value[-1:]), value)], 2, 2, False),
    )
    actor = _FakeActor()
    handle = RayWorkerActorHandle(actor, memory_capacity_bytes=15, worker_id="worker-0")

    def task(query_id):
        return _FakeTask(
            name=f"{query_id}-task-0",
            context={"query_id": query_id, "node_id": "8"},
            inputs={"3": {"kind": "exchange_source_task", "data": b"p0"}},
            plan={"plan": "exchange-template"},
        )

    blocker = handle.submit_tasks([task("query-publication-blocker")])[0]
    assert handle.submit_tasks([task("query-publication")]) == []
    watcher_registration_entered = threading.Event()
    release_watcher_registration = threading.Event()

    def block_watcher_registration(*_args, **_kwargs):
        watcher_registration_entered.set()
        assert release_watcher_registration.wait(5.0)

    monkeypatch.setattr(
        handle,
        "_start_fte_attempt_status_watcher",
        block_watcher_registration,
    )
    local_drop_entered = threading.Event()
    release_local_drop = threading.Event()
    original_drop_registry = task_control_mod._drop_fte_registry_for_query

    def block_local_registry_drop(query_id):
        local_drop_entered.set()
        assert release_local_drop.wait(5.0)
        return original_drop_registry(query_id)

    monkeypatch.setattr(
        task_control_mod,
        "_drop_fte_registry_for_query",
        block_local_registry_drop,
    )
    release_errors = []

    def release_capacity():
        try:
            handle.record_fte_task_terminal(blocker.task_id)
        except BaseException as exc:  # pragma: no cover - asserted below
            release_errors.append(exc)

    release_thread = threading.Thread(target=release_capacity)
    release_thread.start()
    assert watcher_registration_entered.wait(1.0)

    drop_done = threading.Event()
    drop_results = []

    def drop_query():
        try:
            drop_results.append(handle.fte_drop_query("query-publication"))
        except BaseException as exc:  # pragma: no cover - asserted below
            drop_results.append(exc)
        finally:
            drop_done.set()

    drop_thread = threading.Thread(target=drop_query)
    drop_thread.start()
    drop_completed_before_publication = drop_done.wait(0.1)
    release_watcher_registration.set()
    release_thread.join(2.0)
    try:
        assert local_drop_entered.wait(1.0)
        with worker_handle_mod._FTE_REGISTRY_LOCK:
            retained_handles = list(worker_handle_mod._FTE_RESULT_HANDLES_BY_QUERY.get("query-publication", []))
    finally:
        release_local_drop.set()
    drop_thread.join(2.0)

    assert drop_completed_before_publication is False
    assert release_thread.is_alive() is False
    assert drop_thread.is_alive() is False
    assert release_errors == []
    assert [str(result_handle.task_id) for result_handle in retained_handles] == ["query-publication.0.0.0"]
    assert drop_results == [{"tasks_removed": 1, "tasks_canceled": 0, "fragments_removed": 2}]
    assert handle.pop_fte_result_handles("query-publication") == []


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

    retries = low_memory.mark_fte_worker_failed(
        "worker-0",
        RuntimeError("worker lost"),
        worker_incarnation_id=failed_worker.worker_incarnation_id,
    )

    assert len(first) == 1
    assert len(retries) == 1
    assert retries[0].worker_handle is low_memory
    assert worker_handle_mod._FTE_PARTITION_OWNERS[("query-memory-retry", "query-memory-retry:node:7", 0)] is low_memory
    retry_creates = [call for call in actor2.fte_calls if call[0] == "create"]
    assert len(retry_creates) == 1
    assert retry_creates[0][1]["memory_requirement_bytes"] == 10
    assert low_memory.fte_pressure_stats()["assigned_memory_bytes"] == 10
    assert high_memory.fte_pressure_stats()["assigned_memory_bytes"] == 0


def test_fte_worker_failure_does_not_invert_attempt_start_lock_order(monkeypatch):
    failure_holds_registry = threading.Event()
    start_holds_state = threading.Event()
    failure_reads_state_without_registry = threading.Event()
    query_id = "query-worker-failure-start-race"
    fragment_id = f"{query_id}:node:7"
    failed = RayWorkerActorHandle(
        _FakeActor(),
        memory_capacity_bytes=1 << 60,
        worker_id="failed-start-race",
    )
    replacement = RayWorkerActorHandle(
        _FakeActor(),
        memory_capacity_bytes=1 << 60,
        worker_id="replacement-start-race",
    )
    stage = None

    def select_replacement(_partition):
        assert stage is not None
        assert stage._state_lock_owned_by_current_thread()
        start_holds_state.set()
        assert failure_holds_registry.wait(2.0)
        with fte_fragment_scheduler_mod._FTE_REGISTRY_LOCK:
            fte_fragment_scheduler_mod._FTE_PARTITION_OWNERS[owner_key] = replacement
            return replacement.worker_id, replacement

    stage = FteFragmentExecution(
        query_id,
        0,
        fragment_id=fragment_id,
        worker_selector=select_replacement,
        task_memory_bytes=64,
    )
    partition = stage.add_partition(0)
    owner_key = (query_id, fragment_id, 0)

    class _BlockingOwnerRegistry(dict):
        def items(self):
            is_owned = getattr(fte_fragment_scheduler_mod._FTE_REGISTRY_LOCK, "_is_owned", None)
            assert callable(is_owned) and is_owned()
            failure_holds_registry.set()
            assert start_holds_state.wait(2.0)
            return super().items()

    monkeypatch.setattr(
        fte_fragment_scheduler_mod,
        "_FTE_PARTITION_OWNERS",
        _BlockingOwnerRegistry({owner_key: failed}),
    )
    monkeypatch.setitem(
        fte_fragment_scheduler_mod._FTE_FRAGMENT_EXECUTIONS,
        (query_id, fragment_id),
        stage,
    )
    original_has_running_attempt = stage.has_retryable_running_attempt_on_worker
    original_requirements = stage.partition_scheduling_requirements

    def observed_has_running_attempt(*args, **kwargs):
        is_owned = getattr(fte_fragment_scheduler_mod._FTE_REGISTRY_LOCK, "_is_owned", None)
        assert callable(is_owned) and not is_owned()
        failure_reads_state_without_registry.set()
        return original_has_running_attempt(*args, **kwargs)

    def observed_requirements(partition_id):
        is_owned = getattr(fte_fragment_scheduler_mod._FTE_REGISTRY_LOCK, "_is_owned", None)
        assert callable(is_owned) and not is_owned()
        failure_reads_state_without_registry.set()
        return original_requirements(partition_id)

    monkeypatch.setattr(stage, "has_retryable_running_attempt_on_worker", observed_has_running_attempt)
    monkeypatch.setattr(stage, "partition_scheduling_requirements", observed_requirements)

    with ThreadPoolExecutor(max_workers=2) as executor:
        failure = executor.submit(
            fte_fragment_scheduler_mod._mark_fte_worker_failed,
            failed.worker_id,
            {
                "error_code": "WORKER_LOST",
                "message": "worker lost during attempt start",
            },
            query_id_filter=query_id,
            manager_instance_id=failed.manager_instance_id,
            worker_incarnation_id=failed.worker_incarnation_id,
        )
        assert failure_holds_registry.wait(2.0)
        attempt = executor.submit(stage.start_attempt_with_worker, partition)

        scheduled = attempt.result(timeout=2.0)
        assert failure.result(timeout=2.0) == []

    assert failure_reads_state_without_registry.is_set()
    assert scheduled.worker_id == replacement.worker_id
    assert partition.running_attempt is not None
    assert partition.running_attempt.remote_handle is replacement
    assert fte_fragment_scheduler_mod._FTE_PARTITION_OWNERS[owner_key] is replacement


def test_fte_worker_failure_replays_descriptor_on_new_owner(monkeypatch):
    monkeypatch.setattr(
        RayWorkerActorHandle,
        "_fte_task_handle_cls",
        staticmethod(lambda: _FakeFteTaskHandle),
    )
    actor0 = _FakeActor()
    actor1 = _FakeActor()
    kill_calls = []
    monkeypatch.setattr(worker_handle_mod.ray, "kill", lambda actor: kill_calls.append(actor))
    handle0 = RayWorkerActorHandle(actor0, memory_capacity_bytes=1 << 60, worker_id="worker-0")
    handle1 = RayWorkerActorHandle(actor1, memory_capacity_bytes=1 << 60, worker_id="worker-1")
    task = _FakeTask(
        name="scan-task",
        context={"query_id": "query-fte-worker-lost", "node_id": "7"},
        inputs={"7": {"kind": "scan_task", "data": b"a"}},
        plan={"plan": "scan-template"},
    )

    first = handle0.submit_tasks([task])
    retries = handle1.mark_fte_worker_failed(
        "worker-0",
        RuntimeError("actor died"),
        worker_incarnation_id=handle0.worker_incarnation_id,
    )

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
    assert actor0.shutdown_calls == ["prepare"]
    assert kill_calls == [actor0]
    assert "worker-0" not in worker_handle_mod._FTE_WORKER_HANDLES
    stats = handle1.fte_registry_stats()["event_schedulers"]["query-fte-worker-lost"]
    assert stats["event_counts"] == {
        "SplitEventsSubmitted": 1,
        "WorkerReservationCompleted": 2,
        "WorkerFailed": 1,
    }


def test_manager_shutdown_defers_primary_actor_kill_until_finish(monkeypatch):
    monkeypatch.setattr(
        RayWorkerActorHandle,
        "_fte_task_handle_cls",
        staticmethod(lambda: _FakeFteTaskHandle),
    )
    actor0 = _FakeActor()
    actor1 = _FakeActor()
    kill_calls = []
    monkeypatch.setattr(worker_handle_mod.ray, "kill", lambda actor: kill_calls.append(actor))
    handle0 = RayWorkerActorHandle(actor0, memory_capacity_bytes=1 << 60, worker_id="worker-0")
    RayWorkerActorHandle(actor1, memory_capacity_bytes=1 << 60, worker_id="worker-1")
    task = _FakeTask(
        name="copy-task",
        context={
            "query_id": "query-manager-worker-shutdown",
            "node_id": "copy",
            "copy_output_base": "s3://bucket/output",
            "copy_output_run_id": "run-manager-shutdown",
        },
        inputs={"7": {"kind": "scan_task", "data": b"a"}},
        plan={"plan": "copy-template"},
    )
    handle0.submit_tasks([task])

    handle0.prepare_shutdown()

    assert actor0.shutdown_calls == ["prepare", "prepare"]
    assert [call for call in actor1.fte_calls if call[0] == "create"]
    assert kill_calls == []


def test_fte_worker_failure_waits_for_worker_quiescence_before_retry(monkeypatch):
    monkeypatch.setattr(
        RayWorkerActorHandle,
        "_fte_task_handle_cls",
        staticmethod(lambda: _FakeFteTaskHandle),
    )
    quiescence = Future()
    quiescence_started = threading.Event()

    class _DeferredPrepare:
        def remote(self):
            quiescence_started.set()
            return SimpleNamespace(future=lambda: quiescence)

    actor0 = _FakeActor()
    actor0.prepare_shutdown = _DeferredPrepare()
    actor1 = _FakeActor()
    kill_calls = []
    monkeypatch.setattr(worker_handle_mod.ray, "kill", lambda actor: kill_calls.append(actor))
    handle0 = RayWorkerActorHandle(actor0, memory_capacity_bytes=1 << 60, worker_id="worker-0")
    handle1 = RayWorkerActorHandle(actor1, memory_capacity_bytes=1 << 60, worker_id="worker-1")
    task = _FakeTask(
        name="copy-task",
        context={
            "query_id": "query-copy-worker-lost",
            "node_id": "copy",
            "copy_output_base": "s3://bucket/output",
            "copy_output_run_id": "run-1",
        },
        inputs={"7": {"kind": "scan_task", "data": b"a"}},
        plan={"plan": "copy-template"},
    )
    first = handle0.submit_tasks([task])
    retries = []
    errors = []

    def mark_failed():
        try:
            retries.extend(
                handle1.mark_fte_worker_failed(
                    "worker-0",
                    RuntimeError("status RPC failed"),
                    worker_incarnation_id=handle0.worker_incarnation_id,
                )
            )
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    failure_thread = threading.Thread(target=mark_failed)
    failure_thread.start()
    assert quiescence_started.wait(timeout=1.0)

    assert failure_thread.is_alive()
    assert [call for call in actor1.fte_calls if call[0] == "create"] == []
    assert kill_calls == []
    stage = worker_handle_mod._FTE_FRAGMENT_EXECUTIONS[("query-copy-worker-lost", "query-copy-worker-lost:node:copy")]
    assert stage.partitions[0].running_attempt.worker_id == "worker-0"
    with worker_handle_mod._FTE_REGISTRY_LOCK:
        worker_handle_mod._FTE_PARTITION_OWNERS.pop(
            ("query-copy-worker-lost", "query-copy-worker-lost:node:copy", 0),
            None,
        )

    quiescence.set_result(None)
    failure_thread.join(timeout=2.0)

    assert not failure_thread.is_alive()
    assert errors == []
    assert len(first) == 1
    assert len(retries) == 1
    assert retries[0].worker_handle is handle1
    assert [call for call in actor1.fte_calls if call[0] == "create"]
    assert kill_calls == [actor0]


def test_fte_worker_failure_without_confirmed_quiescence_fails_closed(monkeypatch):
    monkeypatch.setattr(
        RayWorkerActorHandle,
        "_fte_task_handle_cls",
        staticmethod(lambda: _FakeFteTaskHandle),
    )
    quiescence = Future()
    quiescence_started = threading.Event()

    class _FailingPrepare:
        def remote(self):
            quiescence_started.set()
            return SimpleNamespace(future=lambda: quiescence)

    actor0 = _FakeActor()
    actor0.prepare_shutdown = _FailingPrepare()
    actor1 = _FakeActor()
    kill_calls = []
    monkeypatch.setattr(worker_handle_mod.ray, "kill", lambda actor: kill_calls.append(actor))
    handle0 = RayWorkerActorHandle(actor0, memory_capacity_bytes=1 << 60, worker_id="worker-0")
    handle1 = RayWorkerActorHandle(actor1, memory_capacity_bytes=1 << 60, worker_id="worker-1")
    task = _FakeTask(
        name="copy-task",
        context={
            "query_id": "query-copy-quiescence-failed",
            "node_id": "copy",
            "copy_output_base": "s3://bucket/output",
            "copy_output_run_id": "run-2",
        },
        inputs={"7": {"kind": "scan_task", "data": b"a"}},
        plan={"plan": "copy-template"},
    )
    first = handle0.submit_tasks([task])
    errors = []

    def mark_failed():
        try:
            handle1.mark_fte_worker_failed(
                "worker-0",
                RuntimeError("status RPC failed"),
                worker_incarnation_id=handle0.worker_incarnation_id,
            )
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    failure_thread = threading.Thread(target=mark_failed)
    failure_thread.start()
    assert quiescence_started.wait(timeout=1.0)

    owner_key = (
        "query-copy-quiescence-failed",
        "query-copy-quiescence-failed:node:copy",
        0,
    )
    stage = worker_handle_mod._FTE_FRAGMENT_EXECUTIONS[owner_key[:2]]
    with worker_handle_mod._FTE_REGISTRY_LOCK:
        worker_handle_mod._FTE_PARTITION_OWNERS.pop(owner_key, None)
    with stage._state_lock:
        stage.partitions[0].running_attempts.clear()
    quiescence.set_exception(RuntimeError("worker side effects still active"))
    failure_thread.join(timeout=2.0)

    assert not failure_thread.is_alive()
    assert len(errors) == 1
    assert "failed to quiesce FTE worker worker-0 before retry" in str(errors[0])
    assert len(first) == 1
    assert [call for call in actor1.fte_calls if call[0] == "create"] == []
    scheduler = worker_handle_mod._FTE_SCHEDULERS.get("query-copy-quiescence-failed")
    assert scheduler is not None
    assert scheduler.stats().state == "FAILED"
    assert "worker-0" not in worker_handle_mod._FTE_WORKER_HANDLES
    assert kill_calls == [actor0]
    assert stage.partitions[0].running_attempts == {}


def test_fte_worker_failure_accepts_confirmed_actor_death_as_quiescence(monkeypatch):
    monkeypatch.setattr(
        RayWorkerActorHandle,
        "_fte_task_handle_cls",
        staticmethod(lambda: _FakeFteTaskHandle),
    )
    quiescence = Future()
    quiescence.set_exception(ray.exceptions.ActorDiedError())

    class _DeadActorPrepare:
        def remote(self):
            return SimpleNamespace(future=lambda: quiescence)

    actor0 = _FakeActor()
    actor0.prepare_shutdown = _DeadActorPrepare()
    actor1 = _FakeActor()
    handle0 = RayWorkerActorHandle(actor0, memory_capacity_bytes=1 << 60, worker_id="worker-0")
    handle1 = RayWorkerActorHandle(actor1, memory_capacity_bytes=1 << 60, worker_id="worker-1")
    task = _FakeTask(
        name="copy-task",
        context={
            "query_id": "query-copy-worker-dead",
            "node_id": "copy",
            "copy_output_base": "s3://bucket/output",
            "copy_output_run_id": "run-3",
        },
        inputs={"7": {"kind": "scan_task", "data": b"a"}},
        plan={"plan": "copy-template"},
    )
    first = handle0.submit_tasks([task])

    retries = handle1.mark_fte_worker_failed(
        "worker-0",
        RuntimeError("actor died"),
        worker_incarnation_id=handle0.worker_incarnation_id,
    )

    assert len(first) == 1
    assert len(retries) == 1
    assert retries[0].worker_handle is handle1
    assert "worker-0" not in worker_handle_mod._FTE_WORKER_HANDLES


def test_fte_worker_failure_event_uses_confirmed_actor_death_without_prepare(monkeypatch):
    monkeypatch.setattr(
        RayWorkerActorHandle,
        "_fte_task_handle_cls",
        staticmethod(lambda: _FakeFteTaskHandle),
    )
    actor0 = _FakeActor()
    actor0.prepare_shutdown = None
    actor1 = _FakeActor()
    handle0 = RayWorkerActorHandle(actor0, memory_capacity_bytes=1 << 60, worker_id="worker-0")
    handle1 = RayWorkerActorHandle(actor1, memory_capacity_bytes=1 << 60, worker_id="worker-1")
    task = _FakeTask(
        name="copy-task",
        context={
            "query_id": "query-copy-worker-confirmed-dead",
            "node_id": "copy",
            "copy_output_base": "s3://bucket/output",
            "copy_output_run_id": "run-4",
        },
        inputs={"7": {"kind": "scan_task", "data": b"a"}},
        plan={"plan": "copy-template"},
    )
    first = handle0.submit_tasks([task])

    retries = handle1._handles_for_worker_failed_event(
        WorkerFailed(
            query_id="query-copy-worker-confirmed-dead",
            worker_id="worker-0",
            worker_incarnation_id=handle0.worker_incarnation_id,
            manager_instance_id=handle0.manager_instance_id,
            error=ray.exceptions.ActorDiedError(),
        )
    )

    assert len(first) == 1
    assert len(retries) == 1
    assert retries[0].worker_handle is handle1
    assert "worker-0" not in worker_handle_mod._FTE_WORKER_HANDLES


def test_worker_quiescence_requires_terminal_actor_death():
    assert fte_fragment_scheduler_mod._worker_actor_death_confirms_quiescence(ray.exceptions.ActorDiedError())
    assert not fte_fragment_scheduler_mod._worker_actor_death_confirms_quiescence(
        ray.exceptions.ActorUnavailableError("actor temporarily unavailable", b"\0" * 16)
    )


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
    retryable_partition.start_attempt(
        worker_id="failed#0",
        worker_incarnation_id=failed.worker_incarnation_id,
        remote_handle=failed,
    )
    non_retryable_partition.start_attempt(
        worker_id="failed#0",
        worker_incarnation_id=failed.worker_incarnation_id,
        remote_handle=failed,
    )
    worker_handle_mod._FTE_PARTITION_OWNERS[(query_id, fragment_id, 0)] = failed
    worker_handle_mod._FTE_PARTITION_OWNERS[(query_id, fragment_id, 1)] = failed

    handles = replacement.mark_fte_worker_failed(
        "failed#0",
        RuntimeError("actor died"),
        worker_incarnation_id=failed.worker_incarnation_id,
    )

    assert stage.partitions[0].failed is False
    assert stage.partitions[1].failed is True
    assert stage.failed is True
    assert all(str(handle.task_id) != f"{query_id}.0.1.1" for handle in handles)


def test_fte_stale_worker_failure_only_reconciles_matching_incarnation():
    query_id = "query-fte-worker-incarnation"
    fragment_id = f"{query_id}:node:7"
    worker_id = "manager-a:node-a:0"
    failed = RayWorkerActorHandle(
        _FakeActor(),
        memory_capacity_bytes=1 << 60,
        worker_id=worker_id,
        manager_instance_id="manager-a",
    )
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
    failed_partition = stage.add_partition(0)
    failed_partition.start_attempt(
        worker_id=worker_id,
        worker_incarnation_id=failed.worker_incarnation_id,
        remote_handle=failed,
    )
    with worker_handle_mod._FTE_REGISTRY_LOCK:
        worker_handle_mod._FTE_WORKER_HANDLES.pop(worker_id)
    replacement = RayWorkerActorHandle(
        _FakeActor(),
        memory_capacity_bytes=1 << 60,
        worker_id=worker_id,
        manager_instance_id="manager-a",
    )
    replacement_partition = stage.add_partition(1)
    replacement_partition.start_attempt(
        worker_id=worker_id,
        worker_incarnation_id=replacement.worker_incarnation_id,
        remote_handle=replacement,
    )

    scheduled = stage.mark_worker_failed(
        worker_id,
        {"error_code": "WORKER_LOST", "message": "old incarnation failed"},
        worker_incarnation_id=failed.worker_incarnation_id,
        retryable=False,
        schedule_retries=False,
    )

    assert scheduled == []
    assert failed_partition.failed is True
    assert replacement_partition.failed is False
    assert replacement_partition.running_attempt is not None
    assert replacement_partition.running_attempt.remote_handle is replacement


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

    retries = handle1.mark_fte_worker_failed(
        "worker-0",
        RuntimeError("actor died"),
        worker_incarnation_id=handle0.worker_incarnation_id,
    )

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


def test_fte_control_quiescence_failure_does_not_skip_later_worker_retirement():
    class _FailingPrepareActor(_FakeActor):
        def _prepare_shutdown(self):
            self.shutdown_calls.append("prepare")
            raise RuntimeError("worker side effects still active")

    first_actor = _FailingPrepareActor()
    second_actor = _FakeActor()
    first = RayWorkerActorHandle(
        first_actor,
        memory_capacity_bytes=1 << 60,
        worker_id="manager-a:node-a:0",
        manager_instance_id="manager-a",
    )
    second = RayWorkerActorHandle(
        second_actor,
        memory_capacity_bytes=1 << 60,
        worker_id="manager-a:node-b:0",
        manager_instance_id="manager-a",
    )
    failures = [
        fte_execution_mod.FteWorkerControlFailure(
            worker_id=worker.worker_id,
            attempt_id=FteTaskAttemptId.coerce(f"query-control-retire-{index}.0.0.0"),
            method_name="fte_create_task",
            cause=RuntimeError(f"planned control failure {index}"),
            worker_incarnation_id=worker.worker_incarnation_id,
        )
        for index, worker in enumerate((first, second))
    ]
    schedulers = []
    for failure in failures:
        scheduler = worker_handle_mod._FTE_SCHEDULERS.get_or_create(failure.attempt_id.task_id.query_id)
        first._bind_fte_scheduler_handlers(scheduler)
        schedulers.append(scheduler)

    for failure in failures:
        first._handles_for_fte_worker_control_failure(failure)

    assert first_actor.shutdown_calls == ["prepare"]
    assert second_actor.shutdown_calls == ["prepare"]
    assert first.worker_id not in worker_handle_mod._FTE_WORKER_HANDLES
    assert second.worker_id not in worker_handle_mod._FTE_WORKER_HANDLES
    assert "failed to quiesce FTE worker" in str(schedulers[0].stats().failure_reason)


def test_fte_control_failure_preempts_queued_work_across_queries(monkeypatch):
    monkeypatch.setattr(
        RayWorkerActorHandle,
        "_fte_task_handle_cls",
        staticmethod(lambda: _FakeFteTaskHandle),
    )

    class _FailFirstCreateActor(_FakeActor):
        failed = False

        def _fte_create_task(self, request):
            self.fte_calls.append(("create", request))
            if request["task_id"]["query_id"] == "query-control-barrier" and not self.failed:
                self.failed = True
                replacement._fte_healthy = True
                raise RuntimeError("planned first create failure")
            return self._control_status("fte_create_task", request["task_id"])

    actor = _FailFirstCreateActor()
    handle = RayWorkerActorHandle(actor, memory_capacity_bytes=1 << 60, worker_id="worker-0")
    replacement_actor = _FakeActor()
    replacement = RayWorkerActorHandle(
        replacement_actor,
        memory_capacity_bytes=1 << 60,
        worker_id="worker-1",
    )
    replacement._fte_healthy = False
    other_handles = handle.submit_tasks(
        [
            _FakeTask(
                name="scan-other",
                context={"query_id": "query-control-other", "node_id": "6"},
                inputs={"6": {"kind": "scan_task", "data": b"other"}},
            )
        ]
    )
    tasks = [
        _FakeTask(
            name="scan-a",
            context={"query_id": "query-control-barrier", "node_id": "7"},
            inputs={"7": {"kind": "scan_task", "data": b"a"}},
        ),
        _FakeTask(
            name="scan-b",
            context={"query_id": "query-control-barrier", "node_id": "8"},
            inputs={"8": {"kind": "scan_task", "data": b"b"}},
        ),
    ]

    handles = handle.submit_tasks(tasks)

    assert len(other_handles) == 1
    assert other_handles[0].worker_handle is handle
    assert len(handles) == 3
    assert all(task_handle.worker_handle is replacement for task_handle in handles)
    assert [str(FteTaskAttemptId.coerce(request["task_id"])) for request in _create_requests(actor)] == [
        "query-control-other.0.0.0",
        "query-control-barrier.0.0.0",
    ]
    assert [str(FteTaskAttemptId.coerce(request["task_id"])) for request in _create_requests(replacement_actor)] == [
        "query-control-other.0.0.1",
        "query-control-barrier.0.0.1",
        "query-control-barrier.1.0.0",
    ]
    assert handle._fte_healthy is False
    assert "worker-0" not in worker_handle_mod._FTE_WORKER_HANDLES
    query_owners = {
        key: owner
        for key, owner in worker_handle_mod._FTE_PARTITION_OWNERS.items()
        if key[0] in {"query-control-barrier", "query-control-other"}
    }
    assert set(query_owners) == {
        ("query-control-barrier", "query-control-barrier:node:7", 0),
        ("query-control-barrier", "query-control-barrier:node:8", 0),
        ("query-control-other", "query-control-other:node:6", 0),
    }
    assert all(owner is replacement for owner in query_owners.values())


def test_fte_split_queue_full_recovers_without_replacing_worker(monkeypatch):
    monkeypatch.setattr(
        RayWorkerActorHandle,
        "_fte_task_handle_cls",
        staticmethod(lambda: _FakeFteTaskHandle),
    )

    class _RecoveringSplitQueueActor(_FakeActor):
        def __init__(self):
            super().__init__()
            self.space_checks = 0

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
            self.space_checks += 1
            return {
                "has_space": self.space_checks >= 2,
                "buffered_splits": 0 if self.space_checks >= 2 else 1024,
                "status": {"state": FteTaskState.RUNNING.value, "task_id": task_id},
            }

    actor0 = _RecoveringSplitQueueActor()
    actor1 = _FakeActor()
    handle0 = RayWorkerActorHandle(actor0, memory_capacity_bytes=1 << 60, worker_id="worker-0")
    _handle1 = RayWorkerActorHandle(actor1, memory_capacity_bytes=1 << 60, worker_id="worker-1")
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
    assert retries == []
    assert [call[0] for call in actor0.fte_calls] == [
        "create",
        "wait_split_queue",
        "wait_split_queue",
        "add_splits",
    ]
    assert [call for call in actor1.fte_calls if call[0] == "create"] == []
    assert handle0._fte_healthy is True
    assert "worker-0" in worker_handle_mod._FTE_WORKER_HANDLES
    assert (
        worker_handle_mod._FTE_PARTITION_OWNERS[("query-fte-queue-full", "query-fte-queue-full:node:7", 0)] is handle0
    )
    stats = handle0.fte_registry_stats()["event_schedulers"]["query-fte-queue-full"]
    assert stats["event_counts"].get("WorkerFailed", 0) == 0


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
    downstream_descriptor = duckdb.ray_cxx.make_exchange_source_task_descriptor_for_test(
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

    retries = handle1.mark_fte_worker_failed(
        "worker-0",
        RuntimeError("host lost"),
        worker_incarnation_id=handle0.worker_incarnation_id,
    )

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
    source_handles = duckdb.ray_cxx.exchange_source_task_source_handles_for_test(
        exchange_retry["initial_splits"]["3"][0]["data"]
    )
    assert source_handles[0]["attempt_id"] == 1
    assert "__attempt_0" not in source_handles[0]["files"][0]["path"]

    assert actor1.register_payloads == [
        [
            {
                "fragment_id": "query-host-loss:node:exchange",
                "plan": {"plan": "exchange-template"},
                "query_id": "query-host-loss",
            }
        ],
        [
            {
                "fragment_id": "query-host-loss:node:scan",
                "plan": {"plan": "scan-template"},
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
    retries = handle.mark_fte_worker_failed(
        "worker-alone",
        RuntimeError("host lost"),
        worker_incarnation_id=handle.worker_incarnation_id,
    )

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
    downstream_descriptor = duckdb.ray_cxx.make_exchange_source_task_descriptor_for_test(
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
        source_handles = duckdb.ray_cxx.exchange_source_task_source_handles_for_test(split["data"])
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


def test_fte_hash_fragment_accepts_one_sided_exchange_updates(monkeypatch):
    monkeypatch.setattr(
        RayWorkerActorHandle,
        "_fte_task_handle_cls",
        staticmethod(lambda: _FakeFteTaskHandle),
    )
    actor = _FakeActor()
    handle = RayWorkerActorHandle(actor, memory_capacity_bytes=1 << 60)
    first_task = _FakeTask(
        name="hash-join-exchange-initial",
        context={"query_id": "query-fte-hash-join-fan-in", "node_id": "8"},
        inputs={
            "3": {
                "kind": "exchange_source_task",
                "data": {
                    "partition_indices": [0],
                    "source_partition_count": 1,
                    "source_task_count": 1,
                    "source_handles": [{"partition_id": 0, "path": "left-sink-a"}],
                },
            },
            "4": {
                "kind": "exchange_source_task",
                "data": {
                    "partition_indices": [0],
                    "source_partition_count": 1,
                    "source_task_count": 1,
                    "source_handles": [{"partition_id": 0, "path": "right-sink-a"}],
                },
            },
        },
        plan={"plan": "hash-join-template"},
    )
    left_update = _FakeTask(
        name="hash-join-exchange-left-update",
        context={"query_id": "query-fte-hash-join-fan-in", "node_id": "8"},
        inputs={
            "3": {
                "kind": "exchange_source_task",
                "data": {
                    "partition_indices": [0],
                    "source_partition_count": 1,
                    "source_task_count": 1,
                    "source_handles": [{"partition_id": 0, "path": "left-sink-b"}],
                },
            }
        },
        plan={"plan": "hash-join-template"},
    )

    first_handles = handle.submit_tasks([first_task])
    actor.fte_calls.clear()
    update_handles = handle.submit_tasks([left_update])

    assert len(first_handles) == 1
    assert update_handles == []
    assert [call[0] for call in actor.fte_calls] == [
        "wait_split_queue",
        "add_splits",
    ]
    add_call = actor.fte_calls[1]
    assert add_call[2] == "3"
    assert add_call[3][0]["data"]["source_handles"][0]["path"] == "left-sink-b"

    actor.fte_calls.clear()
    exhausted_handles = handle.task_input_stream_exhausted(["3", "4"])

    assert exhausted_handles == []
    no_more_calls = [call for call in actor.fte_calls if call[0] == "no_more_splits"]
    assert len(no_more_calls) == 2
    assert {call[2] for call in no_more_calls} == {"3", "4"}


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
    assert query_status["finished"] is False
    fragment_status = next(iter(query_status["fragment_executions"].values()))
    assert fragment_status["finished_count"] == 1
    assert fragment_status["no_more_partitions"] is False
    assert handle.fte_registry_stats()["status_watcher_count"] == 0


def test_fte_status_handler_keeps_watcher_until_terminal_status(monkeypatch):
    from duckdb.runners.ray import fragment_worker_events as worker_events_mod

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

        def handle_task_status(self, status, *, schedule_retry=True):
            del schedule_retry
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
    monkeypatch.setattr(handle, "_drain_fte_pending_tasks", lambda **_kwargs: [])
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
    from duckdb.runners.ray.fte_scheduler import (
        FteAttemptStatusWatcher,
        FteSchedulerRegistry,
    )

    entered = threading.Event()

    class _SlowWorker:
        worker_id = "worker-slow-watcher-drop"
        worker_incarnation_id = "incarnation-slow-watcher-drop"
        manager_instance_id = "manager-a"

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
    from duckdb.runners.ray.fragment_registry import _FteWorkerPressure
    from duckdb.runners.ray.fragment_worker_pressure import partition_reservation_key

    pressure = _FteWorkerPressure()
    parent_attempt = "q.0.0.0"
    child_attempt = "q.child.0.0.0"
    parent_reservation = partition_reservation_key("q", "q:node:1", 0)
    child_reservation = partition_reservation_key("q|child", "q|child:node:1", 0)
    pressure.running_attempts.update({parent_attempt, child_attempt})
    pressure.terminal_attempt_id_by_task.update({"q.0.0": 0, "q.child.0.0": 0})
    pressure.split_counts_by_attempt.update({parent_attempt: 1, child_attempt: 2})
    pressure.pending_split_counts_by_attempt.update({parent_attempt: 3, child_attempt: 4})
    pressure.reserved_partitions.update({parent_reservation, child_reservation})
    pressure.memory_bytes_by_reservation.update({parent_reservation: 10, child_reservation: 20})

    pressure.drop_query("q")

    assert pressure.running_attempts == {child_attempt}
    assert pressure.terminal_attempt_id_by_task == {"q.child.0.0": 0}
    assert pressure.split_counts_by_attempt == {child_attempt: 2}
    assert pressure.pending_split_counts_by_attempt == {child_attempt: 4}
    assert pressure.reserved_partitions == {child_reservation}
    assert pressure.memory_bytes_by_reservation == {child_reservation: 20}


def test_fte_status_watcher_rejects_mismatched_status_identity():
    from duckdb.runners.ray.fte_scheduler import (
        FteAttemptStatusWatcher,
        FteEventHandlers,
        FteSchedulerRegistry,
    )

    class _MismatchedWorker:
        worker_id = "worker-mismatched-watcher-status"
        worker_incarnation_id = "incarnation-mismatched-watcher-status"
        manager_instance_id = "manager-a"

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
    from duckdb.runners.ray import fragment_worker_events as worker_events_mod
    from duckdb.runners.ray.fte_scheduler import FteAttemptStatusWatcher

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
        worker_incarnation_id = "incarnation-close-terminal-race"
        manager_instance_id = "manager-a"

        def fte_wait_task_status(self, task_id, _min_version, _timeout_s):
            return {
                "state": "FAILED",
                "task_id": task_id,
                "failure": {
                    "error_code": "GENERIC_INTERNAL_ERROR",
                    "message": "retryable",
                },
                "version": 1,
            }

    class _BlockedFragmentExecution:
        def handle_task_status(self, _status, *, schedule_retry=True):
            del schedule_retry
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


def test_fte_terminal_query_drop_race_still_drains_other_queries(monkeypatch):
    monkeypatch.setattr(
        fragment_submission_mod,
        "_split_exchange_source_task_by_partition",
        lambda value: ([(int(value[-1:]), value)], 2, 2, False),
    )
    actor = _FakeActor()
    handle = RayWorkerActorHandle(actor, memory_capacity_bytes=15, worker_id="worker-0")

    def task(query_id):
        return _FakeTask(
            name=f"{query_id}-task-0",
            context={"query_id": query_id, "node_id": "8"},
            inputs={"3": {"kind": "exchange_source_task", "data": b"p0"}},
            plan={"plan": "exchange-template"},
        )

    query_a_handles = handle.submit_tasks([task("query-a")])
    query_b_handles = handle.submit_tasks([task("query-b")])

    assert [str(task_handle.task_id) for task_handle in query_a_handles] == ["query-a.0.0.0"]
    assert query_b_handles == []
    assert [str(task_handle.task_id) for task_handle in handle.pop_fte_result_handles("query-a")] == ["query-a.0.0.0"]

    fragment_execution = worker_handle_mod._FTE_FRAGMENT_EXECUTIONS[("query-a", "query-a:node:8")]
    original_handle_task_status = fragment_execution.handle_task_status
    mutation_done = threading.Event()
    release_handler = threading.Event()

    def blocked_handle_task_status(status, *, schedule_retry=True):
        result = original_handle_task_status(status, schedule_retry=schedule_retry)
        mutation_done.set()
        assert release_handler.wait(2.0)
        return result

    monkeypatch.setattr(fragment_execution, "handle_task_status", blocked_handle_task_status)
    completion_handles = []
    completion_errors = []

    def finish_query_a():
        try:
            completion_handles.extend(
                handle.handle_fte_task_status(
                    {
                        "state": "FINISHED",
                        "task_id": query_a_handles[0].task_id.to_dict(),
                        "version": 1,
                    }
                )
            )
        except BaseException as exc:  # pragma: no cover - asserted below
            completion_errors.append(exc)

    completion_thread = threading.Thread(target=finish_query_a)
    completion_thread.start()
    assert mutation_done.wait(1.0)

    drop_result = handle.fte_drop_query("query-a")
    release_handler.set()
    completion_thread.join(2.0)

    assert completion_thread.is_alive() is False
    assert completion_errors == []
    assert drop_result == {"tasks_removed": 1, "tasks_canceled": 0, "fragments_removed": 2}
    # Query B was admitted by the teardown-owned pump, not returned through
    # the already-closing query A event.
    assert completion_handles == []
    scheduled_query_b = handle.pop_fte_result_handles("query-b")
    assert [str(task_handle.task_id) for task_handle in scheduled_query_b] == ["query-b.0.0.0"]


def test_fte_terminal_mutation_error_still_drains_other_queries(monkeypatch):
    monkeypatch.setattr(
        fragment_submission_mod,
        "_split_exchange_source_task_by_partition",
        lambda value: ([(int(value[-1:]), value)], 2, 2, False),
    )
    actor = _FakeActor()
    handle = RayWorkerActorHandle(actor, memory_capacity_bytes=15, worker_id="worker-0")

    def task(query_id):
        return _FakeTask(
            name=f"{query_id}-task-0",
            context={"query_id": query_id, "node_id": "8"},
            inputs={"3": {"kind": "exchange_source_task", "data": b"p0"}},
            plan={"plan": "exchange-template"},
        )

    query_a_handles = handle.submit_tasks([task("query-a")])
    assert handle.submit_tasks([task("query-b")]) == []
    assert [str(task_handle.task_id) for task_handle in handle.pop_fte_result_handles("query-a")] == ["query-a.0.0.0"]

    fragment_execution = worker_handle_mod._FTE_FRAGMENT_EXECUTIONS[("query-a", "query-a:node:8")]

    def fail_descriptor_remove(_task_id):
        raise OSError("descriptor spill unlink failed")

    monkeypatch.setattr(fragment_execution.descriptor_storage, "remove", fail_descriptor_remove)
    with pytest.raises(OSError, match="descriptor spill unlink failed"):
        handle.handle_fte_task_status(
            {
                "state": "FINISHED",
                "task_id": query_a_handles[0].task_id.to_dict(),
                "version": 1,
            }
        )

    assert [str(task_handle.task_id) for task_handle in handle.pop_fte_result_handles("query-b")] == ["query-b.0.0.0"]


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
            "failure": {
                "error_code": "GENERIC_INTERNAL_ERROR",
                "message": "retry me",
            },
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
            "failure": {
                "error_code": "GENERIC_INTERNAL_ERROR",
                "message": "not retryable",
                "retryable": False,
            },
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
    script = """
import ray

from duckdb.runners.ray import worker as worker_mod

actor_cls = worker_mod.RayWorkerActor.__ray_metadata__.modified_class
payload = ray.cloudpickle.dumps(actor_cls)
restored = ray.cloudpickle.loads(payload)
assert restored.__name__ == actor_cls.__name__
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr or result.stdout


def test_ray_worker_fte_debug_logs_use_worker_topology(monkeypatch, capsys):
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
    monkeypatch.setenv("VANE_FTE_RESULT_DEBUG", "1")
    monkeypatch.setenv("VANE_WORKER_ID", "ray-worker-log")
    monkeypatch.setenv("VANE_WORKER_MANAGER_INSTANCE_ID", "manager-log")
    monkeypatch.setenv("VANE_WORKER_NODE_ID", "node-log")
    monkeypatch.setenv("VANE_WORKER_HOST", "10.0.0.9")

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
    assert "manager_instance_id=manager-log" in captured
    assert "node_id=node-log" in captured
    assert "host=10.0.0.9" in captured
    assert "event=manager_init" in captured
    assert "event=create_task" in captured
    assert "event=start_task" in captured
    assert "event=task_done" in captured
    assert "task_id=ray-log.0.0.0" in captured
    assert "max_running=4" in captured
    result_lines = [line for line in captured.splitlines() if "[vane-fte-result" in line]
    assert result_lines
    assert all("worker_id=ray-worker-log" in line for line in result_lines)
    assert all("manager_instance_id=manager-log" in line for line in result_lines)
    assert all("node_id=node-log" in line for line in result_lines)
    assert all("host=10.0.0.9" in line for line in result_lines)


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
    replay_registrations = []
    monkeypatch.setattr(
        worker_mod,
        "_register_query_python_replay_state",
        lambda query_id, plan: replay_registrations.append((query_id, plan)) or True,
    )

    class _Plan:
        @staticmethod
        def resource_query_id():
            return "query-resource"

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
    assert replay_registrations == [("query-resource", resolved_plan)]


def test_start_ray_workers_keeps_flight_host_worker_local_and_skips_nested_warmup(monkeypatch):
    get_calls = []
    option_calls = []
    remote_calls = []

    class _FakePingMethod:
        def remote(self, *_args, **_kwargs):
            return "warmup-ref"

    class _FakeActorHandle:
        def __init__(self):
            self.ping = _FakePingMethod()

    class _FakeActorFactory:
        def options(self, **kwargs):
            option_calls.append(kwargs)
            return self

        def remote(self, **kwargs):
            remote_calls.append(kwargs)
            return _FakeActorHandle()

    monkeypatch.setattr(worker_handle_mod, "_is_ray_worker_context", lambda: True)
    monkeypatch.setattr(
        worker_handle_mod,
        "_collect_vane_env_overrides",
        lambda: {"VANE_FLIGHT_ADVERTISE_HOST": "flight.example.internal"},
    )
    monkeypatch.setattr(worker_handle_mod, "RayWorkerActor", _FakeActorFactory())
    monkeypatch.setattr(
        worker_handle_mod.ray,
        "nodes",
        lambda: [
            {
                "NodeID": "node-a",
                "NodeManagerAddress": "10.0.0.1",
                "Resources": {"CPU": 4.0, "memory": 1024.0, "GPU": 0.0},
            },
            {
                "NodeID": "node-b",
                "NodeManagerAddress": "10.0.0.2",
                "Resources": {"CPU": 4.0, "memory": 1024.0, "GPU": 0.0},
            },
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

    runtimes = worker_handle_mod.start_ray_workers(
        existing_worker_ids=[],
        manager_instance_id="manager-a",
    )
    other_manager_runtimes = worker_handle_mod.start_ray_workers(
        existing_worker_ids=[],
        manager_instance_id="manager-b",
    )
    refreshed_runtimes = worker_handle_mod.start_ray_workers(
        existing_worker_ids=["manager-a:node-a:0", "manager-a:node-b:0"],
        manager_instance_id="manager-a",
    )

    assert len(runtimes) == 2
    assert len(other_manager_runtimes) == 2
    assert refreshed_runtimes == []
    assert sorted(worker_handle_mod._FTE_WORKER_HANDLES) == [
        "manager-a:node-a:0",
        "manager-a:node-b:0",
        "manager-b:node-a:0",
        "manager-b:node-b:0",
    ]
    for index, (node_id, address) in enumerate((("node-a", "10.0.0.1"), ("node-b", "10.0.0.2"))):
        worker_id = f"manager-a:{node_id}:0"
        assert option_calls[index]["memory"] == 358
        assert option_calls[index]["runtime_env"] == {
            "env_vars": {
                "RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO": "0",
                "VANE_WORKER": "1",
                "VANE_WORKER_HOST": address,
                "VANE_WORKER_ID": worker_id,
                "VANE_WORKER_INDEX": "0",
                "VANE_WORKER_MANAGER_INSTANCE_ID": "manager-a",
                "VANE_WORKER_NODE_ID": node_id,
            },
        }
        assert "num_cpus" not in option_calls[index]
        assert "num_gpus" not in option_calls[index]
        assert "env_overrides" not in remote_calls[index]
        assert remote_calls[index]["duckdb_memory_bytes"] == 256
        assert remote_calls[index]["task_heap_capacity_bytes"] == 615
        assert remote_calls[index]["ray_node_ip_address"] == address
        assert worker_handle_mod._FTE_WORKER_HANDLES[worker_id].host == address
        assert worker_handle_mod._FTE_WORKER_HANDLES[worker_id].manager_instance_id == "manager-a"
        other_worker_id = f"manager-b:{node_id}:0"
        assert option_calls[index + 2]["runtime_env"]["env_vars"]["VANE_WORKER_ID"] == other_worker_id
        assert worker_handle_mod._FTE_WORKER_HANDLES[other_worker_id].host == address
        assert worker_handle_mod._FTE_WORKER_HANDLES[other_worker_id].manager_instance_id == "manager-b"
    assert get_calls == []


@pytest.mark.parametrize(
    ("warmup_error", "error_match"),
    [
        (RuntimeError("planned warmup failure"), "planned warmup failure"),
        (ray.exceptions.GetTimeoutError("planned warmup timeout"), "Failed to warm up Worker actors within 120s"),
    ],
)
def test_start_ray_workers_cleans_up_actors_after_warmup_failure(monkeypatch, warmup_error, error_match):
    killed = []

    class _FakePingMethod:
        def remote(self):
            return "warmup-ref"

    class _FakeActorHandle:
        def __init__(self, node_id):
            self.node_id = node_id
            self.ping = _FakePingMethod()

    class _FakeActorFactory:
        def __init__(self):
            self.node_id = None

        def options(self, **kwargs):
            self.node_id = kwargs["scheduling_strategy"][1]["node_id"]
            return self

        def remote(self, **_kwargs):
            return _FakeActorHandle(self.node_id)

    monkeypatch.setattr(worker_handle_mod, "_is_ray_worker_context", lambda: False)
    monkeypatch.setattr(worker_handle_mod, "_collect_vane_env_overrides", dict)
    monkeypatch.setattr(worker_handle_mod, "RayWorkerActor", _FakeActorFactory())
    monkeypatch.setattr(
        worker_handle_mod.ray,
        "nodes",
        lambda: [
            {
                "NodeID": node_id,
                "NodeManagerAddress": address,
                "Resources": {"CPU": 4.0, "memory": 1024.0, "GPU": 0.0},
            }
            for node_id, address in (("node-a", "10.0.0.1"), ("node-b", "10.0.0.2"))
        ],
    )
    monkeypatch.setattr(
        worker_handle_mod.ray.util.scheduling_strategies,
        "NodeAffinitySchedulingStrategy",
        lambda **kwargs: ("node-affinity", kwargs),
    )
    monkeypatch.setattr(
        worker_handle_mod,
        "resolve_object_refs_blocking",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(warmup_error),
    )
    monkeypatch.setattr(
        worker_handle_mod.ray,
        "kill",
        lambda actor, *, no_restart=True: killed.append((actor.node_id, no_restart)),
    )

    with pytest.raises(RuntimeError, match=error_match):
        worker_handle_mod.start_ray_workers(
            existing_worker_ids=[],
            manager_instance_id="manager-a",
        )

    assert sorted(killed) == [("node-a", True), ("node-b", True)]


def test_worker_session_connection_rejects_reopen_after_close(monkeypatch):
    actor_cls = worker_mod.RayWorkerActor.__ray_metadata__.modified_class
    actor = object.__new__(actor_cls)
    actor._session_connections = {}
    actor._session_operation_locks = {}
    actor._closed_session_ids = worker_mod.BoundedReplayMap(capacity=65_536)
    actor._session_connections_lock = threading.RLock()
    actor._shutdown_started = False
    closed = []

    class _SessionConnection:
        def close(self):
            closed.append("session")

    class _SharedConnection:
        def cursor(self):
            return _SessionConnection()

    actor._get_shared_conn = lambda: _SharedConnection()

    def _configure(_connection, config, *, use_session_credentials):
        assert use_session_credentials is True
        return dict(config)

    monkeypatch.setattr(worker_mod, "_configure_duckdb_s3", _configure)

    session_connection = actor_cls._get_session_conn(
        actor,
        "session-a",
        {"AWS_ACCESS_KEY_ID": "key-a"},
    )
    assert (
        actor_cls._get_session_conn(
            actor,
            "session-a",
            {"AWS_ACCESS_KEY_ID": "key-a"},
        )
        is session_connection
    )

    asyncio.run(actor_cls.close_session(actor, "session-a"))

    assert closed == ["session"]
    with pytest.raises(RuntimeError, match="Vane session is closed"):
        actor_cls._get_session_conn(
            actor,
            "session-a",
            {"AWS_ACCESS_KEY_ID": "key-a"},
        )
    assert actor._session_operation_locks == {}


def test_worker_handle_session_close_treats_dead_actor_as_terminal(monkeypatch):
    close_ref = object()
    calls = []

    class _CloseMethod:
        @staticmethod
        def remote(session_id):
            calls.append(session_id)
            return close_ref

    worker_handle = object.__new__(_ProductionRayWorkerActorHandle)
    worker_handle.actor_handle = SimpleNamespace(close_session=_CloseMethod())

    import duckdb.runners.ray.safe_get as safe_get

    monkeypatch.setattr(
        safe_get,
        "resolve_object_refs_blocking",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ray.exceptions.RayActorError(error_msg="worker exited")),
    )

    worker_handle.close_session("session-a")

    assert calls == ["session-a"]


def test_worker_session_connection_open_is_single_flight(monkeypatch):
    actor_cls = worker_mod.RayWorkerActor.__ray_metadata__.modified_class
    actor = object.__new__(actor_cls)
    actor._session_connections = {}
    actor._session_s3_configs = {}
    actor._session_operation_locks = {}
    actor._closed_session_ids = worker_mod.BoundedReplayMap(capacity=65_536)
    actor._session_connections_lock = threading.RLock()
    actor._shutdown_started = False
    configure_started = threading.Event()
    configure_release = threading.Event()
    created_connections = []
    results = []
    errors = []

    class _SessionConnection:
        def close(self):
            raise AssertionError("single-flight session connection must not be discarded")

    class _SharedConnection:
        def cursor(self):
            connection = _SessionConnection()
            created_connections.append(connection)
            return connection

    def _configure(_connection, config, *, use_session_credentials):
        assert use_session_credentials is True
        configure_started.set()
        assert configure_release.wait(timeout=1.0)
        return dict(config)

    actor._get_shared_conn = lambda: _SharedConnection()
    monkeypatch.setattr(worker_mod, "_configure_duckdb_s3", _configure)

    def _open():
        try:
            results.append(actor_cls._get_session_conn(actor, "session-a", {"AWS_PROFILE": "analytics"}))
        except BaseException as exc:
            errors.append(exc)

    first = threading.Thread(target=_open)
    second = threading.Thread(target=_open)
    first.start()
    assert configure_started.wait(timeout=1.0)
    second.start()
    time.sleep(0.01)
    assert len(created_connections) == 1
    configure_release.set()
    first.join(timeout=1.0)
    second.join(timeout=1.0)

    assert not first.is_alive()
    assert not second.is_alive()
    assert errors == []
    assert results == [created_connections[0], created_connections[0]]


def test_worker_session_credential_refresh_is_single_flight(monkeypatch):
    actor_cls = worker_mod.RayWorkerActor.__ray_metadata__.modified_class
    actor = object.__new__(actor_cls)
    connection = object()
    actor._session_connections = {"session-a": ({"AWS_PROFILE": "analytics"}, connection)}
    actor._session_s3_configs = {
        "session-a": {
            "AWS_ACCESS_KEY_ID": "old-key",
            "AWS_SECRET_ACCESS_KEY": "old-secret",
            worker_mod._AWS_CREDENTIAL_REFRESH_AT_KEY: "0",
        }
    }
    actor._session_operation_locks = {}
    actor._closed_session_ids = worker_mod.BoundedReplayMap(capacity=65_536)
    actor._session_connections_lock = threading.RLock()
    actor._shutdown_started = False
    resolver_started = threading.Event()
    resolver_release = threading.Event()
    resolver_calls = []
    results = []
    errors = []

    def _resolve(config):
        resolver_calls.append(dict(config))
        resolver_started.set()
        assert resolver_release.wait(timeout=1.0)
        return (
            {
                "AWS_ACCESS_KEY_ID": "new-key",
                "AWS_SECRET_ACCESS_KEY": "new-secret",
            },
            200.0,
        )

    monkeypatch.setattr(worker_mod, "_resolve_session_aws_credentials", _resolve)
    monkeypatch.setattr(worker_mod.time, "time", lambda: 100.0)

    def _refresh():
        try:
            results.append(
                actor_cls._refresh_session_s3_config(
                    actor,
                    "session-a",
                    {"AWS_PROFILE": "analytics"},
                    connection,
                    use_session_credentials=True,
                )
            )
        except BaseException as exc:
            errors.append(exc)

    first = threading.Thread(target=_refresh)
    second = threading.Thread(target=_refresh)
    first.start()
    assert resolver_started.wait(timeout=1.0)
    second.start()
    time.sleep(0.01)
    assert len(resolver_calls) == 1
    resolver_release.set()
    first.join(timeout=1.0)
    second.join(timeout=1.0)

    assert not first.is_alive()
    assert not second.is_alive()
    assert errors == []
    assert len(resolver_calls) == 1
    assert [result["AWS_ACCESS_KEY_ID"] for result in results] == ["new-key", "new-key"]


def test_worker_session_configuration_failure_closes_unpublished_connection(monkeypatch):
    actor_cls = worker_mod.RayWorkerActor.__ray_metadata__.modified_class
    actor = object.__new__(actor_cls)
    actor._session_connections = {}
    actor._closed_session_ids = worker_mod.BoundedReplayMap(capacity=65_536)
    actor._session_connections_lock = threading.RLock()
    actor._shutdown_started = False
    closed = []

    class _SessionConnection:
        def close(self):
            closed.append("session")

    class _SharedConnection:
        def cursor(self):
            return _SessionConnection()

    actor._get_shared_conn = lambda: _SharedConnection()

    def _fail_configuration(_connection, _config, *, use_session_credentials):
        assert use_session_credentials is True
        raise RuntimeError("planned session configuration failure")

    monkeypatch.setattr(
        worker_mod,
        "_configure_duckdb_s3",
        _fail_configuration,
    )

    with pytest.raises(RuntimeError, match="planned session configuration failure"):
        actor_cls._get_session_conn(
            actor,
            "session-a",
            {"AWS_ACCESS_KEY_ID": "key-a"},
        )

    assert closed == ["session"]
    assert actor._session_connections == {}


def test_worker_session_close_wins_race_with_credential_resolution(monkeypatch):
    actor_cls = worker_mod.RayWorkerActor.__ray_metadata__.modified_class
    actor = object.__new__(actor_cls)
    actor._session_connections = {}
    actor._session_s3_configs = {}
    actor._closed_session_ids = worker_mod.BoundedReplayMap(capacity=65_536)
    actor._session_connections_lock = threading.RLock()
    actor._shutdown_started = False
    resolution_started = threading.Event()
    resolution_release = threading.Event()
    closed = []
    open_errors = []

    class _SessionConnection:
        def close(self):
            closed.append("session")

    class _SharedConnection:
        def cursor(self):
            return _SessionConnection()

    def _delayed_config(_connection, config, *, use_session_credentials):
        assert use_session_credentials is True
        resolution_started.set()
        assert resolution_release.wait(timeout=1.0)
        return dict(config)

    actor._get_shared_conn = lambda: _SharedConnection()
    monkeypatch.setattr(worker_mod, "_configure_duckdb_s3", _delayed_config)

    def _open():
        try:
            actor_cls._get_session_conn(actor, "session-a", {"AWS_PROFILE": "analytics"})
        except BaseException as exc:
            open_errors.append(exc)

    open_thread = threading.Thread(target=_open)
    open_thread.start()
    assert resolution_started.wait(timeout=1.0)

    close_errors = []

    def _close():
        try:
            asyncio.run(actor_cls.close_session(actor, "session-a"))
        except BaseException as exc:
            close_errors.append(exc)

    close_thread = threading.Thread(target=_close)
    close_thread.start()
    for _ in range(100):
        if "session-a" in actor._closed_session_ids:
            break
        time.sleep(0.001)
    assert close_thread.is_alive()
    resolution_release.set()
    open_thread.join(timeout=1.0)
    close_thread.join(timeout=1.0)

    assert not open_thread.is_alive()
    assert not close_thread.is_alive()
    assert close_errors == []
    assert len(open_errors) == 1
    assert isinstance(open_errors[0], RuntimeError)
    assert "Vane session is closed" in str(open_errors[0])
    assert closed == ["session"]
    assert actor._session_connections == {}
    assert actor._session_s3_configs == {}


def test_worker_session_close_wins_race_with_credential_refresh(monkeypatch):
    actor_cls = worker_mod.RayWorkerActor.__ray_metadata__.modified_class
    actor = object.__new__(actor_cls)
    actor._session_operation_locks = {}
    actor._closed_session_ids = worker_mod.BoundedReplayMap(capacity=65_536)
    actor._session_connections_lock = threading.RLock()
    actor._shutdown_started = False
    refresh_started = threading.Event()
    refresh_release = threading.Event()
    closed = []
    refresh_errors = []
    close_errors = []

    class _SessionConnection:
        def close(self):
            closed.append("session")

    connection = _SessionConnection()
    session_config = {"AWS_PROFILE": "analytics"}
    actor._session_connections = {
        "session-a": (
            session_config,
            connection,
        ),
    }
    actor._session_s3_configs = {
        "session-a": {
            "AWS_ACCESS_KEY_ID": "old-key",
            "AWS_SECRET_ACCESS_KEY": "old-secret",
        },
    }

    def _delayed_refresh(_config, _effective, *, use_session_credentials):
        assert use_session_credentials is True
        refresh_started.set()
        assert refresh_release.wait(timeout=1.0)
        return {
            "AWS_ACCESS_KEY_ID": "new-key",
            "AWS_SECRET_ACCESS_KEY": "new-secret",
        }

    monkeypatch.setattr(
        worker_mod,
        "_refresh_effective_duckdb_s3_config",
        _delayed_refresh,
    )

    def _refresh():
        try:
            actor_cls._refresh_session_s3_config(
                actor,
                "session-a",
                session_config,
                connection,
                use_session_credentials=True,
            )
        except BaseException as exc:
            refresh_errors.append(exc)

    refresh_thread = threading.Thread(target=_refresh)
    refresh_thread.start()
    assert refresh_started.wait(timeout=1.0)

    def _close():
        try:
            asyncio.run(actor_cls.close_session(actor, "session-a"))
        except BaseException as exc:
            close_errors.append(exc)

    close_thread = threading.Thread(target=_close)
    close_thread.start()
    for _ in range(100):
        if "session-a" in actor._closed_session_ids:
            break
        time.sleep(0.001)
    assert close_thread.is_alive()

    refresh_release.set()
    refresh_thread.join(timeout=1.0)
    close_thread.join(timeout=1.0)

    assert not refresh_thread.is_alive()
    assert not close_thread.is_alive()
    assert close_errors == []
    assert len(refresh_errors) == 1
    assert "Vane session is closed" in str(refresh_errors[0])
    assert closed == ["session"]
    assert actor._session_connections == {}
    assert actor._session_s3_configs == {}
    assert actor._session_operation_locks == {}


def test_worker_session_close_keeps_connection_retryable_after_failure():
    actor_cls = worker_mod.RayWorkerActor.__ray_metadata__.modified_class
    actor = object.__new__(actor_cls)
    actor._closed_session_ids = worker_mod.BoundedReplayMap(capacity=65_536)
    actor._session_connections_lock = threading.RLock()
    close_calls = []

    class _SessionConnection:
        def close(self):
            close_calls.append("close")
            if len(close_calls) == 1:
                raise RuntimeError("planned close failure")

    connection = _SessionConnection()
    record = ({}, connection)
    actor._session_connections = {"session-a": record}

    with pytest.raises(RuntimeError, match="planned close failure"):
        asyncio.run(actor_cls.close_session(actor, "session-a"))

    assert actor._session_connections["session-a"] is record
    assert "session-a" in actor._closed_session_ids

    asyncio.run(actor_cls.close_session(actor, "session-a"))

    assert close_calls == ["close", "close"]
    assert "session-a" not in actor._session_connections


def test_worker_session_close_does_not_block_actor_event_loop():
    actor_cls = worker_mod.RayWorkerActor.__ray_metadata__.modified_class
    actor = object.__new__(actor_cls)
    actor._closed_session_ids = worker_mod.BoundedReplayMap(capacity=65_536)
    actor._session_connections_lock = threading.RLock()
    close_started = threading.Event()
    close_release = threading.Event()

    class _SessionConnection:
        def close(self):
            close_started.set()
            assert close_release.wait(timeout=1.0)

    actor._session_connections = {"session-a": ({}, _SessionConnection())}

    async def _close_without_blocking_loop():
        close_task = asyncio.create_task(actor_cls.close_session(actor, "session-a"))
        assert await asyncio.to_thread(close_started.wait, 1.0)
        await asyncio.wait_for(asyncio.sleep(0), timeout=0.1)
        assert close_task.done() is False
        close_release.set()
        await asyncio.wait_for(close_task, timeout=1.0)

    asyncio.run(_close_without_blocking_loop())

    assert "session-a" not in actor._session_connections


def test_worker_session_close_cancellation_waits_for_owned_teardown():
    actor_cls = worker_mod.RayWorkerActor.__ray_metadata__.modified_class
    actor = object.__new__(actor_cls)
    actor._closed_session_ids = worker_mod.BoundedReplayMap(capacity=65_536)
    actor._session_connections_lock = threading.RLock()
    close_started = threading.Event()
    close_release = threading.Event()

    class _SessionConnection:
        def close(self):
            close_started.set()
            assert close_release.wait(timeout=1.0)

    actor._session_connections = {"session-a": ({}, _SessionConnection())}

    async def _cancel_close():
        close_task = asyncio.create_task(actor_cls.close_session(actor, "session-a"))
        assert await asyncio.to_thread(close_started.wait, 1.0)
        close_task.cancel()
        await asyncio.sleep(0.01)
        assert close_task.done() is False
        close_release.set()
        with pytest.raises(asyncio.CancelledError):
            await close_task

    asyncio.run(_cancel_close())

    assert "session-a" not in actor._session_connections


def test_execute_native_task_configuration_failure_closes_unregistered_cursor(monkeypatch):
    actor_cls = worker_mod.RayWorkerActor.__ray_metadata__.modified_class
    actor = object.__new__(actor_cls)
    actor._session_connections_lock = threading.RLock()
    actor._session_s3_configs = {}
    actor._session_operation_locks = {}
    actor._closed_session_ids = worker_mod.BoundedReplayMap(capacity=65_536)
    actor._shutdown_started = False
    actor._native_execution_condition = threading.Condition()
    actor._active_native_cursors = set()
    actor._native_cursor_query_ids = {}
    actor._closing_native_queries = set()
    closed = []

    class _Cursor:
        def close(self):
            closed.append("cursor")

    cursor = _Cursor()
    session_conn = SimpleNamespace(cursor=lambda: cursor)
    actor._session_connections = {
        "session-a": (
            {
                "AWS_ACCESS_KEY_ID": "key-a",
                "AWS_SECRET_ACCESS_KEY": "secret-a",
            },
            session_conn,
        ),
    }

    def _get_session_conn(session_id, config, *, use_session_credentials):
        assert session_id == "session-a"
        assert config == {
            "AWS_ACCESS_KEY_ID": "key-a",
            "AWS_SECRET_ACCESS_KEY": "secret-a",
        }
        assert use_session_credentials is True
        return session_conn

    actor._get_session_conn = _get_session_conn

    def _fail_configuration(_connection, _config, *, use_session_credentials):
        assert use_session_credentials is True
        raise RuntimeError("planned query cursor configuration failure")

    monkeypatch.setattr(
        worker_mod,
        "_configure_duckdb_s3",
        _fail_configuration,
    )

    class _Plan:
        @staticmethod
        def session_id():
            return "session-a"

        @staticmethod
        def session_config():
            return {
                "AWS_ACCESS_KEY_ID": "key-a",
                "AWS_SECRET_ACCESS_KEY": "secret-a",
            }

        @staticmethod
        def has_explicit_s3_credentials():
            return False

    with pytest.raises(RuntimeError, match="planned query cursor configuration failure"):
        actor_cls._execute_native_task(actor, _Plan(), None, native_query_id="query-a")

    assert closed == ["cursor"]
    assert actor._active_native_cursors == set()
    assert actor._native_cursor_query_ids == {}
    assert getattr(actor, "_native_query_cleanup_contexts", {}) == {}


def test_execute_native_task_passes_exchange_and_sink_inputs(monkeypatch):
    actor_cls = worker_mod.RayWorkerActor.__ray_metadata__.modified_class
    actor = object.__new__(actor_cls)
    actor._session_connections_lock = threading.RLock()
    actor._session_s3_configs = {}
    actor._session_operation_locks = {}
    actor._closed_session_ids = worker_mod.BoundedReplayMap(capacity=65_536)
    actor._shutdown_started = False
    actor._native_execution_condition = threading.Condition()
    actor._active_native_cursors = set()
    actor._native_cursor_query_ids = {}
    actor._closing_native_queries = set()
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
            effective_session_config,
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
                    effective_session_config,
                )
            )
            return "ok"

    class _FakePlan:
        def session_id(self):
            return "session-a"

        def session_config(self):
            return {
                "AWS_ACCESS_KEY_ID": "session-a-key",
                "AWS_SECRET_ACCESS_KEY": "session-a-secret",
            }

        def has_explicit_s3_credentials(self):
            return False

    shared_conn = _FakeConn()
    actor._session_connections = {
        "session-a": (
            {
                "AWS_ACCESS_KEY_ID": "session-a-key",
                "AWS_SECRET_ACCESS_KEY": "session-a-secret",
            },
            shared_conn,
        ),
    }

    def _get_session_conn(session_id, config, *, use_session_credentials):
        assert (session_id, config) == (
            "session-a",
            {
                "AWS_ACCESS_KEY_ID": "session-a-key",
                "AWS_SECRET_ACCESS_KEY": "session-a-secret",
            },
        )
        assert use_session_credentials is True
        return shared_conn

    actor._get_session_conn = _get_session_conn
    actor._get_plan_runner = lambda: _FakePlanRunner()
    plan_object = _FakePlan()
    configured = []

    def _configure(cursor, config, *, use_session_credentials):
        configured.append((cursor, dict(config), use_session_credentials))
        return dict(config)

    monkeypatch.setattr(worker_mod, "_configure_duckdb_s3", _configure)

    dynamic_domains = {"df0": {"column": "id", "single_value": 7}}
    result = actor_cls._execute_native_task(
        actor,
        plan_object,
        {"1": b"scan"},
        copy_output_info={"base": "", "run_id": "run-native", "remote_base": "/tmp/out"},
        exchange_source_task_map={"9": b"exchange-binding"},
        exchange_sink_instance={"sink_handle": {"partition_id": 4}, "attempt_id": 2, "attempt_path": "/tmp/attempt"},
        dynamic_filter_domains=dynamic_domains,
        debug_context={"query_id": "q1", "fragment_id": "f1", "task_id": "q1.2.3.4"},
    )

    assert result == "ok"
    assert configured == [
        (
            shared_conn.cursor_obj,
            {
                "AWS_ACCESS_KEY_ID": "session-a-key",
                "AWS_SECRET_ACCESS_KEY": "session-a-secret",
            },
            True,
        )
    ]
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
        effective_session_config,
    ) = calls[0]
    assert plan is plan_object
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
    assert effective_session_config == {
        "AWS_ACCESS_KEY_ID": "session-a-key",
        "AWS_SECRET_ACCESS_KEY": "session-a-secret",
    }


def test_configure_duckdb_s3_applies_static_credentials_only_to_connection_context():
    statements = []

    class _FakeConnection:
        def execute(self, statement):
            statements.append(statement)

    worker_mod._configure_duckdb_s3(
        _FakeConnection(),
        {
            "AWS_ACCESS_KEY_ID": "access-key",
            "AWS_SECRET_ACCESS_KEY": "secret'value",
        },
    )

    assert statements[0] == "LOAD httpfs"
    assert "SET s3_access_key_id='access-key'" in statements
    assert "SET s3_secret_access_key='secret''value'" in statements
    assert "SET s3_session_token=''" in statements
    assert all("SET GLOBAL" not in statement for statement in statements)


def test_configure_duckdb_s3_does_not_install_httpfs_when_load_fails():
    statements = []

    class _FakeConnection:
        def execute(self, statement):
            statements.append(statement)
            raise RuntimeError("httpfs is unavailable")

    with pytest.raises(RuntimeError, match="runtime extension installation is disabled") as exc_info:
        worker_mod._configure_duckdb_s3(
            _FakeConnection(),
            {
                "AWS_ACCESS_KEY_ID": "access-key",
                "AWS_SECRET_ACCESS_KEY": "secret-key",
            },
        )

    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert statements == ["LOAD httpfs"]


def test_configure_duckdb_s3_preserves_scheme_less_endpoint_authority():
    statements = []

    class _FakeConnection:
        def execute(self, statement):
            statements.append(statement)

    worker_mod._configure_duckdb_s3(
        _FakeConnection(),
        {"AWS_ENDPOINT_URL": "minio.internal:9000"},
    )

    assert "SET s3_endpoint='minio.internal:9000'" in statements
    assert "SET s3_use_ssl=false" in statements


@pytest.mark.parametrize(
    "provider_config",
    [
        {"AWS_PROFILE": "analytics"},
        {
            "AWS_ROLE_ARN": "arn:aws:iam::123456789012:role/analytics",
            "AWS_WEB_IDENTITY_TOKEN_FILE": "/var/run/secrets/aws/token",
        },
    ],
)
def test_configure_duckdb_s3_resolves_session_credential_chain_outside_shared_worker(
    monkeypatch,
    provider_config,
):
    statements = []
    resolver_calls = []

    class _FakeConnection:
        def execute(self, statement):
            statements.append(statement)

    def _resolve(config):
        resolver_calls.append(dict(config))
        return (
            {
                "AWS_ACCESS_KEY_ID": "resolved-key",
                "AWS_SECRET_ACCESS_KEY": "resolved-secret",
                "AWS_SESSION_TOKEN": "resolved-token",
                "AWS_REGION": "us-west-2",
            },
            None,
        )

    monkeypatch.setattr(worker_mod, "_resolve_session_aws_credentials", _resolve)

    effective_config = worker_mod._configure_duckdb_s3(
        _FakeConnection(),
        provider_config,
    )

    assert resolver_calls == [provider_config]
    assert effective_config == {
        "AWS_ACCESS_KEY_ID": "resolved-key",
        "AWS_SECRET_ACCESS_KEY": "resolved-secret",
        "AWS_SESSION_TOKEN": "resolved-token",
        "AWS_REGION": "us-west-2",
    }
    assert "SET s3_access_key_id='resolved-key'" in statements
    assert "SET s3_secret_access_key='resolved-secret'" in statements
    assert "SET s3_session_token='resolved-token'" in statements
    assert "SET s3_region='us-west-2'" in statements


@pytest.mark.parametrize(
    "partial_credentials",
    [
        {"AWS_ACCESS_KEY_ID": "partial-key"},
        {"AWS_SECRET_ACCESS_KEY": "partial-secret"},
        {"AWS_SESSION_TOKEN": "partial-token"},
    ],
)
def test_session_credentials_reject_incomplete_static_values(monkeypatch, partial_credentials):
    resolver_calls = []
    monkeypatch.setattr(
        worker_mod,
        "_resolve_session_aws_credentials",
        lambda config: resolver_calls.append(dict(config)),
    )

    with pytest.raises(ValueError, match="must provide both"):
        worker_mod._effective_duckdb_s3_config(partial_credentials)

    assert resolver_calls == []


def test_session_credential_resolver_uses_only_explicit_child_environment(monkeypatch):
    monkeypatch.setenv("AWS_PROFILE", "shared-process-profile")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "shared-process-key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "shared-process-secret")
    calls = []

    def _run(command, **kwargs):
        calls.append((command, kwargs))
        assert kwargs["env"]["AWS_PROFILE"] == "connection-profile"
        assert kwargs["env"]["AWS_WEB_IDENTITY_TOKEN_FILE"] == "/session/token"
        assert "AWS_ACCESS_KEY_ID" not in kwargs["env"]
        assert "AWS_SECRET_ACCESS_KEY" not in kwargs["env"]
        assert kwargs["env"]["VANE_RUNNER"] == "local"
        return SimpleNamespace(
            returncode=0,
            stdout=(
                '{"credentials":{"AWS_ACCESS_KEY_ID":"resolved-key",'
                '"AWS_SECRET_ACCESS_KEY":"resolved-secret",'
                '"AWS_SESSION_TOKEN":"resolved-token"},'
                '"expiration_epoch_s":1234.5}'
            ),
            stderr="",
        )

    monkeypatch.setattr(worker_mod.subprocess, "run", _run)

    credentials, expiration_epoch_s = worker_mod._resolve_session_aws_credentials(
        {
            "AWS_PROFILE": "connection-profile",
            "AWS_WEB_IDENTITY_TOKEN_FILE": "/session/token",
        }
    )

    assert credentials == {
        "AWS_ACCESS_KEY_ID": "resolved-key",
        "AWS_SECRET_ACCESS_KEY": "resolved-secret",
        "AWS_SESSION_TOKEN": "resolved-token",
    }
    assert expiration_epoch_s == 1234.5
    assert calls[0][0] == [sys.executable, "-m", "duckdb.runners.ray.aws_credentials"]
    assert calls[0][1]["capture_output"] is True
    assert calls[0][1]["check"] is False
    assert calls[0][1]["text"] is True
    assert calls[0][1]["timeout"] == 120
    assert os.environ["AWS_PROFILE"] == "shared-process-profile"
    assert os.environ["AWS_ACCESS_KEY_ID"] == "shared-process-key"
    assert os.environ["AWS_SECRET_ACCESS_KEY"] == "shared-process-secret"


def test_session_credential_resolver_loads_profile_in_real_child_process(tmp_path):
    credentials_file = tmp_path / "credentials"
    credentials_file.write_text(
        "[analytics]\n"
        "aws_access_key_id=profile-key\n"
        "aws_secret_access_key=profile-secret\n"
        "aws_session_token=profile-token\n",
        encoding="utf-8",
    )

    credentials, expiration_epoch_s = worker_mod._resolve_session_aws_credentials(
        {
            "AWS_EC2_METADATA_DISABLED": "true",
            "AWS_PROFILE": "analytics",
            "AWS_REGION": "us-east-2",
            "AWS_SHARED_CREDENTIALS_FILE": str(credentials_file),
        }
    )

    assert credentials == {
        "AWS_ACCESS_KEY_ID": "profile-key",
        "AWS_REGION": "us-east-2",
        "AWS_SECRET_ACCESS_KEY": "profile-secret",
        "AWS_SESSION_TOKEN": "profile-token",
    }
    assert expiration_epoch_s is None


def test_session_credential_chain_refreshes_before_expiration(monkeypatch):
    now = [100.0]
    resolver_calls = []

    def _resolve(config):
        resolver_calls.append(dict(config))
        suffix = len(resolver_calls)
        return (
            {
                "AWS_ACCESS_KEY_ID": f"key-{suffix}",
                "AWS_SECRET_ACCESS_KEY": f"secret-{suffix}",
            },
            200.0 + suffix * 100.0,
        )

    monkeypatch.setattr(worker_mod, "_resolve_session_aws_credentials", _resolve)
    monkeypatch.setattr(worker_mod.time, "time", lambda: now[0])

    config = {"AWS_PROFILE": "analytics"}
    effective = worker_mod._effective_duckdb_s3_config(config)
    assert effective["AWS_ACCESS_KEY_ID"] == "key-1"

    now[0] = 259.9
    assert worker_mod._refresh_effective_duckdb_s3_config(config, effective) == effective
    assert len(resolver_calls) == 1

    now[0] = 260.0
    refreshed = worker_mod._refresh_effective_duckdb_s3_config(config, effective)
    assert refreshed["AWS_ACCESS_KEY_ID"] == "key-2"
    assert len(resolver_calls) == 2


def test_nonexpiring_session_credential_chain_is_resolved_once(monkeypatch):
    resolver_calls = []

    def _resolve(config):
        resolver_calls.append(dict(config))
        return (
            {
                "AWS_ACCESS_KEY_ID": "profile-key",
                "AWS_SECRET_ACCESS_KEY": "profile-secret",
            },
            None,
        )

    monkeypatch.setattr(worker_mod, "_resolve_session_aws_credentials", _resolve)
    config = {"AWS_PROFILE": "analytics"}

    effective = worker_mod._refresh_effective_duckdb_s3_config(config, {})
    refreshed = worker_mod._refresh_effective_duckdb_s3_config(config, effective)

    assert refreshed == effective
    assert resolver_calls == [config]


def test_explicit_duckdb_credentials_skip_profile_resolution_and_discard_cached_profile_credentials(monkeypatch):
    resolver_calls = []
    monkeypatch.setattr(
        worker_mod,
        "_resolve_session_aws_credentials",
        lambda config: resolver_calls.append(dict(config)),
    )
    config = {
        "AWS_ACCESS_KEY_ID": "incomplete-environment-key",
        "AWS_PROFILE": "unavailable-profile",
        "AWS_ENDPOINT_URL": "https://s3.example.test",
    }
    cached = {
        "AWS_ACCESS_KEY_ID": "stale-profile-key",
        "AWS_SECRET_ACCESS_KEY": "stale-profile-secret",
        "AWS_SESSION_TOKEN": "stale-profile-token",
        "AWS_ENDPOINT_URL": "https://s3.example.test",
        worker_mod._AWS_CREDENTIAL_REFRESH_AT_KEY: "9999999999",
    }
    statements = []

    class _FakeConnection:
        def execute(self, statement):
            statements.append(statement)

    effective = worker_mod._refresh_effective_duckdb_s3_config(
        config,
        cached,
        use_session_credentials=False,
    )
    configured = worker_mod._configure_duckdb_s3(
        _FakeConnection(),
        effective,
        use_session_credentials=False,
    )

    assert effective == {"AWS_ENDPOINT_URL": "https://s3.example.test"}
    assert configured == effective
    assert "SET s3_access_key_id=''" in statements
    assert "SET s3_secret_access_key=''" in statements
    assert "SET s3_session_token=''" in statements
    assert resolver_calls == []


def test_execute_native_task_uses_session_database_for_fte():
    actor_cls = worker_mod.RayWorkerActor.__ray_metadata__.modified_class
    actor = object.__new__(actor_cls)
    actor._shutdown_started = False
    actor._session_connections_lock = threading.RLock()
    actor._closed_session_ids = worker_mod.BoundedReplayMap(capacity=16)
    actor._session_s3_configs = {}
    actor._native_execution_condition = threading.Condition()
    actor._active_native_cursors = set()
    actor._native_cursor_query_ids = {}
    actor._closing_native_queries = set()
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
    actor._session_connections = {"session-a": ({}, shared_conn)}

    class _FakePlan:
        def session_id(self):
            return "session-a"

        def session_config(self):
            return {}

        def has_explicit_s3_credentials(self):
            return False

    actor._get_session_conn = lambda session_id, config, *, use_session_credentials: (
        shared_conn if (session_id, config) == ("session-a", {}) else None
    )

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
            _effective_session_config,
        ):
            calls.append((cursor, fte_scan_source_queues, fte_exchange_source_queues))
            return "ok"

    shared_runner = _FakePlanRunner()
    actor._get_plan_runner = lambda: shared_runner

    scan_queues = {"1": object()}
    exchange_queues = {"2": object()}
    result = actor_cls._execute_native_task(
        actor,
        _FakePlan(),
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
