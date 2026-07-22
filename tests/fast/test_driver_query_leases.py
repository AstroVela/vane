# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace

import pytest

from vane.runners.ray.query_execution_graph import (
    NodeResourceAllocation,
    QueryAllocation,
    QueryExecutionGraph,
    ResourceVector,
    StageResourceSpec,
)
from vane.runners.ray.query_resource_runtime import (
    clear_query_resource_managers,
    register_query_graph,
)


@pytest.fixture(autouse=True)
def _clean_query_runtime():
    clear_query_resource_managers()
    yield
    clear_query_resource_managers()


def _graph(query_id: str) -> QueryExecutionGraph:
    stage = StageResourceSpec(
        query_id=query_id,
        stage_id=f"stage:{query_id}:udf",
        physical_node_id="node:1:udf",
        stage_kind="udf",
        backend="ray_task",
        input_stage_ids=(),
        per_task=ResourceVector(cpu=1, heap_bytes=100),
        target_output_block_bytes=10,
        generator_buffer_blocks=2,
        max_concurrency=None,
    )
    return QueryExecutionGraph(
        query_id=query_id,
        plan_digest="sha256:test-driver-query-leases",
        stages=(stage,),
        terminal_stage_ids=(stage.stage_id,),
    )


def _allocation() -> QueryAllocation:
    resources = ResourceVector(cpu=1, heap_bytes=101, object_store_bytes=20)
    return QueryAllocation(
        resources=resources,
        node_allocations=(NodeResourceAllocation(node_id="node-a", resources=resources),),
        actor_placements=(),
        generation=1,
    )


def _runner(loop: asyncio.AbstractEventLoop):
    from vane.runners.ray.driver import RayQueryDriverActor

    runner_cls = RayQueryDriverActor.__ray_metadata__.modified_class
    runner = object.__new__(runner_cls)
    runner._query_task_lease_requests = {}
    runner._query_output_lease_requests = {}
    runner._query_task_admission_pumps = set()
    runner._query_output_admission_pumps = set()
    runner._query_resource_admission_loop = loop
    return runner_cls, runner


def _task_request(query_id: str, request_id: str, task_id: str) -> dict:
    return {
        "request_id": request_id,
        "query_id": query_id,
        "stage_id": f"stage:{query_id}:udf",
        "task_id": task_id,
        "attempt_id": f"attempt:{task_id}",
        "node_id": None,
        "retained_input_bytes": 0,
        "resources": {
            "cpu": 1.0,
            "gpu": 0.0,
            "heap_bytes": 100,
            "object_store_bytes": 0,
        },
    }


async def _fill_task_output_window(runner_cls, runner, query_id, stage_id, task, *, prefix):
    owned = []
    for index in range(2):
        request = {
            "request_id": f"request:{prefix}:window:{index}",
            "query_id": query_id,
            "producer_stage_id": stage_id,
            "task_lease_id": task["lease"]["lease_id"],
            "attempt_id": task["lease"]["attempt_id"],
            "block_id": f"block:{prefix}:window:{index}",
            "size_bytes": 10,
        }
        grant = await runner_cls.acquire_query_output_block_lease(runner, request)
        assert grant["granted"] is True
        owned.append((request, grant))
    return owned


async def _release_owned_outputs(runner_cls, runner, owned):
    for request, grant in owned:
        await runner_cls.release_query_output_block_lease(
            runner,
            request["request_id"],
            grant["lease"]["lease_id"],
        )


def test_driver_task_lease_wait_is_event_driven_and_release_wakes_next_request():
    async def scenario():
        query_id = "query-lease-wakeup"
        graph = _graph(query_id)
        runner_cls, runner = _runner(asyncio.get_running_loop())
        manager = register_query_graph(
            graph,
            _allocation(),
            on_change=lambda: runner_cls._signal_query_resource_change(runner, query_id),
        )
        manager.update_stage_state(graph.stages[0].stage_id, runnable=True)

        first_request = _task_request(query_id, "request:first", "task:first")
        first = await runner_cls.acquire_query_task_lease(runner, first_request)
        assert first["granted"]
        first_lease = first["lease"]
        assert await runner_cls.mark_query_task_lease_submitted(
            runner, first_request["request_id"], first_lease["lease_id"]
        ) == {"submitted": True}

        second_request = _task_request(query_id, "request:second", "task:second")
        second_waiter = asyncio.create_task(runner_cls.acquire_query_task_lease(runner, second_request))
        for _ in range(3):
            await asyncio.sleep(0)
        assert not second_waiter.done()

        released = await runner_cls.release_query_task_lease(
            runner,
            first_request["request_id"],
            first_lease["lease_id"],
            first_lease["attempt_id"],
        )
        assert released == {"released": True}
        second_state = runner._query_task_lease_requests[second_request["request_id"]]
        assert second_state["status"] == "granted"
        assert second_state["future"].done()
        second = await asyncio.wait_for(second_waiter, timeout=1)
        assert second["granted"]
        assert second["lease"]["task_id"] == "task:second"

        output = await runner_cls.acquire_query_output_block_lease(
            runner,
            {
                "request_id": "output-request:block-1",
                "query_id": query_id,
                "producer_stage_id": graph.stages[0].stage_id,
                "task_lease_id": second["lease"]["lease_id"],
                "attempt_id": second["lease"]["attempt_id"],
                "block_id": "block-1",
                "size_bytes": 10,
            },
        )
        assert output["granted"]
        assert output["lease"]["state"] == "stage_queue"
        assert await runner_cls.handoff_query_output_block_lease(
            runner,
            "output-request:block-1",
            output["lease"]["lease_id"],
        ) == {"handed_off": True}
        assert manager.snapshot()["output_leases"][output["lease"]["lease_id"]]["state"] == "downstream_input"
        assert await runner_cls.handoff_query_output_block_lease(
            runner,
            "output-request:block-1",
            output["lease"]["lease_id"],
        ) == {"handed_off": False}
        assert await runner_cls.release_query_output_block_lease(
            runner,
            "output-request:block-1",
            output["lease"]["lease_id"],
        ) == {"released": True}
        assert await runner_cls.release_query_task_lease(
            runner,
            second_request["request_id"],
            second["lease"]["lease_id"],
            second["lease"]["attempt_id"],
        ) == {"released": True}
        snapshot = manager.snapshot()
        assert snapshot["task_leases"] == {}
        assert snapshot["output_leases"] == {}
        assert runner._query_task_lease_requests == {}
        assert runner._query_output_lease_requests == {}

    asyncio.run(scenario())


def test_driver_resource_change_evaluates_only_the_selected_task_waiter():
    async def scenario():
        query_id = "query-single-admission-pump"
        graph = _graph(query_id)
        runner_cls, runner = _runner(asyncio.get_running_loop())
        manager = register_query_graph(
            graph,
            _allocation(),
            on_change=lambda: runner_cls._signal_query_resource_change(runner, query_id),
        )
        manager.update_stage_state(graph.stages[0].stage_id, runnable=True)

        active_request = _task_request(query_id, "request:active", "task:active")
        active = await runner_cls.acquire_query_task_lease(runner, active_request)

        pending_requests = [
            _task_request(query_id, f"request:pending:{index}", f"task:pending:{index}") for index in range(6)
        ]
        pending_tasks = [
            asyncio.create_task(runner_cls.acquire_query_task_lease(runner, request)) for request in pending_requests
        ]
        for _ in range(5):
            await asyncio.sleep(0)
        assert manager.snapshot()["stages"][graph.stages[0].stage_id]["pending_task_count"] == 6

        evaluated_task_ids = []
        original_block_reason = manager._normal_task_block_reason_locked

        def counted_block_reason(request, **kwargs):
            evaluated_task_ids.append(request.task_id)
            return original_block_reason(request, **kwargs)

        manager._normal_task_block_reason_locked = counted_block_reason
        await runner_cls.release_query_task_lease(
            runner,
            active_request["request_id"],
            active["lease"]["lease_id"],
            active["lease"]["attempt_id"],
        )
        assert evaluated_task_ids == [f"task:pending:{index}" for index in range(6)]
        done, _ = await asyncio.wait(
            pending_tasks,
            timeout=1,
            return_when=asyncio.FIRST_COMPLETED,
        )
        assert len(done) == 1
        for _ in range(5):
            await asyncio.sleep(0)

        granted_task = next(iter(done))
        granted = granted_task.result()
        for request, task in zip(pending_requests, pending_tasks):
            if task is granted_task:
                continue
            await runner_cls.cancel_query_task_lease_request(
                runner,
                request["request_id"],
                submitted=False,
            )
        await asyncio.gather(*pending_tasks)
        granted_request = pending_requests[pending_tasks.index(granted_task)]
        await runner_cls.release_query_task_lease(
            runner,
            granted_request["request_id"],
            granted["lease"]["lease_id"],
            granted["lease"]["attempt_id"],
        )

    asyncio.run(scenario())


def test_driver_resource_change_event_drives_fte_owner_without_polling(monkeypatch):
    import vane.runners.ray.fte_fragment_scheduler as fte_scheduler

    calls: list[str] = []
    drained = threading.Event()

    monkeypatch.setattr(
        fte_scheduler,
        "has_fte_resource_admission_waiter",
        lambda query_id: query_id == "query-fte-resource-wake",
    )

    def drain(query_id: str):
        calls.append(query_id)
        drained.set()
        return []

    monkeypatch.setattr(
        fte_scheduler,
        "drain_fte_resource_admission_change",
        drain,
    )

    async def scenario():
        runner_cls, runner = _runner(asyncio.get_running_loop())
        runner_cls._signal_query_resource_change(
            runner,
            "query-fte-resource-wake",
        )
        assert await asyncio.to_thread(drained.wait, 1.0)
        for _ in range(20):
            await asyncio.sleep(0)
            task = runner._query_fte_admission_pumps.get("query-fte-resource-wake")
            if task is None or task.done():
                break
        assert calls == ["query-fte-resource-wake"]

    asyncio.run(scenario())


def test_driver_pending_task_lease_can_be_cancelled_without_a_polling_wakeup():
    async def scenario():
        query_id = "query-lease-cancel"
        graph = _graph(query_id)
        runner_cls, runner = _runner(asyncio.get_running_loop())
        manager = register_query_graph(
            graph,
            _allocation(),
            on_change=lambda: runner_cls._signal_query_resource_change(runner, query_id),
        )
        manager.update_stage_state(graph.stages[0].stage_id, runnable=True)

        active_request = _task_request(query_id, "request:active", "task:active")
        active = await runner_cls.acquire_query_task_lease(runner, active_request)
        pending_request = _task_request(query_id, "request:pending", "task:pending")
        pending = asyncio.create_task(runner_cls.acquire_query_task_lease(runner, pending_request))
        for _ in range(3):
            await asyncio.sleep(0)
        assert not pending.done()
        waiting_stage = manager.snapshot()["stages"][graph.stages[0].stage_id]
        assert waiting_stage["pending_task_count"] == 1
        assert waiting_stage["queued_input_bytes"] == 1

        cancelled = await runner_cls.cancel_query_task_lease_request(
            runner,
            pending_request["request_id"],
            submitted=False,
        )
        assert cancelled == {"cancelled": True, "released": False}
        denial = await asyncio.wait_for(pending, timeout=1)
        assert denial["granted"] is False
        assert denial["fatal"] is True
        assert denial["blocked_reason"] == "task_lease_request_cancelled"
        assert manager.snapshot()["stages"][graph.stages[0].stage_id]["pending_task_count"] == 0
        assert pending_request["request_id"] not in runner._query_task_lease_requests
        assert active_request["request_id"] in runner._query_task_lease_requests

        await runner_cls.release_query_task_lease(
            runner,
            active_request["request_id"],
            active["lease"]["lease_id"],
            active["lease"]["attempt_id"],
        )
        assert manager.snapshot()["task_leases"] == {}
        assert runner._query_task_lease_requests == {}

    asyncio.run(scenario())


def test_driver_rejects_runtime_resources_that_diverge_from_registered_stage():
    async def scenario():
        query_id = "query-lease-resource-mismatch"
        graph = _graph(query_id)
        runner_cls, runner = _runner(asyncio.get_running_loop())
        manager = register_query_graph(
            graph,
            _allocation(),
            on_change=lambda: runner_cls._signal_query_resource_change(runner, query_id),
        )
        manager.update_stage_state(graph.stages[0].stage_id, runnable=True)
        request = _task_request(query_id, "request:mismatch", "task:mismatch")
        request["resources"]["heap_bytes"] = 99

        denial = await runner_cls.acquire_query_task_lease(runner, request)

        assert denial["granted"] is False
        assert denial["fatal"] is True
        assert denial["blocked_reason"] == "task_resource_spec_mismatch"
        assert manager.snapshot()["task_leases"] == {}
        assert manager.snapshot()["stages"][graph.stages[0].stage_id]["pending_task_count"] == 0
        assert runner._query_task_lease_requests == {}
        owner_identity = (query_id, "task:mismatch", "attempt:task:mismatch")
        assert owner_identity not in runner._query_task_request_owner_by_identity

        corrected = _task_request(
            query_id,
            "request:mismatch-corrected",
            "task:mismatch",
        )
        corrected_grant = await runner_cls.acquire_query_task_lease(runner, corrected)
        assert corrected_grant["granted"] is True
        assert len(manager.snapshot()["task_leases"]) == 1

    asyncio.run(scenario())


def test_driver_pending_output_waiter_is_live_and_cancel_cleanup_is_complete():
    async def scenario():
        query_id = "query-output-waiter-cancel"
        graph = _graph(query_id)
        runner_cls, runner = _runner(asyncio.get_running_loop())
        manager = register_query_graph(
            graph,
            _allocation(),
            on_change=lambda: runner_cls._signal_query_resource_change(runner, query_id),
        )
        stage_id = graph.stages[0].stage_id
        manager.update_stage_state(stage_id, runnable=True)
        task_request = _task_request(query_id, "request:task", "task:output")
        task = await runner_cls.acquire_query_task_lease(runner, task_request)
        owned = await _fill_task_output_window(
            runner_cls,
            runner,
            query_id,
            stage_id,
            task,
            prefix="cancel",
        )
        output_request = {
            "request_id": "request:output",
            "query_id": query_id,
            "producer_stage_id": stage_id,
            "task_lease_id": task["lease"]["lease_id"],
            "attempt_id": task["lease"]["attempt_id"],
            "block_id": "blocked-output",
            "size_bytes": 10,
        }
        pending = asyncio.create_task(runner_cls.acquire_query_output_block_lease(runner, output_request))
        for _ in range(3):
            await asyncio.sleep(0)
        assert pending.done() is False
        stage_snapshot = manager.snapshot()["stages"][stage_id]
        assert stage_snapshot["pending_output_count"] == 1
        assert stage_snapshot["queued_output_bytes"] == 30

        assert await runner_cls.cancel_query_output_block_lease_request(
            runner,
            output_request["request_id"],
        ) == {"cancelled": True, "released": False}
        denial = await asyncio.wait_for(pending, timeout=1)
        assert denial["blocked_reason"] == "output_lease_request_cancelled"
        assert denial["fatal"] is True
        assert manager.snapshot()["stages"][stage_id]["pending_output_count"] == 0
        assert set(runner._query_output_lease_requests) == {request["request_id"] for request, _grant in owned}

        await _release_owned_outputs(runner_cls, runner, owned)
        assert runner._query_output_lease_requests == {}

        await runner_cls.release_query_task_lease(
            runner,
            task_request["request_id"],
            task["lease"]["lease_id"],
            task["lease"]["attempt_id"],
        )

    asyncio.run(scenario())


def test_driver_output_pump_evaluates_each_waiter_once_per_state_transition():
    async def scenario():
        query_id = "query-single-output-admission-pump"
        graph = _graph(query_id)
        runner_cls, runner = _runner(asyncio.get_running_loop())
        manager = register_query_graph(
            graph,
            _allocation(),
            on_change=lambda: runner_cls._signal_query_resource_change(runner, query_id),
        )
        stage_id = graph.stages[0].stage_id
        manager.update_stage_state(stage_id, runnable=True)
        task_request = _task_request(query_id, "request:task", "task:output-pump")
        task = await runner_cls.acquire_query_task_lease(runner, task_request)
        owned = await _fill_task_output_window(
            runner_cls,
            runner,
            query_id,
            stage_id,
            task,
            prefix="pump",
        )

        output_requests = [
            {
                "request_id": f"request:output:{index}",
                "query_id": query_id,
                "producer_stage_id": stage_id,
                "task_lease_id": task["lease"]["lease_id"],
                "attempt_id": task["lease"]["attempt_id"],
                "block_id": f"blocked-output:{index}",
                "size_bytes": 10,
            }
            for index in range(6)
        ]
        output_tasks = [
            asyncio.create_task(runner_cls.acquire_query_output_block_lease(runner, request))
            for request in output_requests
        ]
        for _ in range(8):
            await asyncio.sleep(0)
        assert all(not task.done() for task in output_tasks)

        original_block_reason = manager._normal_output_block_reason_locked
        evaluated_block_ids = []

        def counted_block_reason(request):
            evaluated_block_ids.append(request.block_id)
            return original_block_reason(request)

        manager._normal_output_block_reason_locked = counted_block_reason
        await runner_cls.release_query_output_block_lease(
            runner,
            owned[0][0]["request_id"],
            owned[0][1]["lease"]["lease_id"],
        )
        for _ in range(10):
            await asyncio.sleep(0)

        assert len([task for task in output_tasks if task.done()]) == 1
        await runner_cls.release_query_output_block_lease(
            runner,
            owned[1][0]["request_id"],
            owned[1][1]["lease"]["lease_id"],
        )
        for _ in range(10):
            await asyncio.sleep(0)

        granted_tasks = [task for task in output_tasks if task.done()]
        assert len(granted_tasks) == 2
        assert set(evaluated_block_ids) == {
            *(f"blocked-output:{index}" for index in range(6)),
        }

        for request, pending in zip(output_requests, output_tasks):
            if pending.done():
                continue
            assert await runner_cls.cancel_query_output_block_lease_request(
                runner,
                request["request_id"],
            ) == {"cancelled": True, "released": False}
        results = await asyncio.gather(*output_tasks)
        for request, result in zip(output_requests, results):
            if result["granted"]:
                assert await runner_cls.release_query_output_block_lease(
                    runner,
                    request["request_id"],
                    result["lease"]["lease_id"],
                ) == {"released": True}
            else:
                assert result["blocked_reason"] == "output_lease_request_cancelled"

        assert await runner_cls.release_query_task_lease(
            runner,
            task_request["request_id"],
            task["lease"]["lease_id"],
            task["lease"]["attempt_id"],
        ) == {"released": True}

    asyncio.run(scenario())


def test_task_attempt_rejects_a_second_request_id_instead_of_orphaning_future():
    async def scenario():
        query_id = "query-duplicate-task-owner"
        graph = _graph(query_id)
        runner_cls, runner = _runner(asyncio.get_running_loop())
        manager = register_query_graph(
            graph,
            _allocation(),
            on_change=lambda: runner_cls._signal_query_resource_change(
                runner,
                query_id,
            ),
        )
        manager.update_stage_state(graph.stages[0].stage_id, runnable=True)
        active_request = _task_request(
            query_id,
            "request:active",
            "task:active",
        )
        active = await runner_cls.acquire_query_task_lease(runner, active_request)
        first_request = _task_request(
            query_id,
            "request:first-owner",
            "task:duplicate",
        )
        first_waiter = asyncio.create_task(runner_cls.acquire_query_task_lease(runner, first_request))
        for _ in range(4):
            await asyncio.sleep(0)
        assert not first_waiter.done()

        duplicate_request = dict(first_request)
        duplicate_request["request_id"] = "request:second-owner"
        with pytest.raises(ValueError, match="task attempt is already owned"):
            await asyncio.wait_for(
                runner_cls.acquire_query_task_lease(runner, duplicate_request),
                timeout=0.1,
            )

        assert await runner_cls.cancel_query_task_lease_request(
            runner,
            first_request["request_id"],
            submitted=False,
        ) == {"cancelled": True, "released": False}
        denial = await first_waiter
        assert denial["blocked_reason"] == "task_lease_request_cancelled"
        assert await runner_cls.release_query_task_lease(
            runner,
            active_request["request_id"],
            active["lease"]["lease_id"],
            active["lease"]["attempt_id"],
        ) == {"released": True}

    asyncio.run(scenario())


def test_output_block_rejects_a_second_request_id_instead_of_orphaning_future():
    async def scenario():
        query_id = "query-duplicate-output-owner"
        graph = _graph(query_id)
        runner_cls, runner = _runner(asyncio.get_running_loop())
        manager = register_query_graph(
            graph,
            _allocation(),
            on_change=lambda: runner_cls._signal_query_resource_change(
                runner,
                query_id,
            ),
        )
        stage_id = graph.stages[0].stage_id
        manager.update_stage_state(stage_id, runnable=True)
        task_request = _task_request(
            query_id,
            "request:task",
            "task:output-owner",
        )
        task = await runner_cls.acquire_query_task_lease(runner, task_request)
        owned = await _fill_task_output_window(
            runner_cls,
            runner,
            query_id,
            stage_id,
            task,
            prefix="duplicate",
        )
        first_request = {
            "request_id": "request:first-output-owner",
            "query_id": query_id,
            "producer_stage_id": stage_id,
            "task_lease_id": task["lease"]["lease_id"],
            "attempt_id": task["lease"]["attempt_id"],
            "block_id": "block:duplicate",
            "size_bytes": 10,
        }
        first_waiter = asyncio.create_task(runner_cls.acquire_query_output_block_lease(runner, first_request))
        for _ in range(4):
            await asyncio.sleep(0)
        assert not first_waiter.done()

        duplicate_request = dict(first_request)
        duplicate_request["request_id"] = "request:second-output-owner"
        with pytest.raises(ValueError, match="output block is already owned"):
            await asyncio.wait_for(
                runner_cls.acquire_query_output_block_lease(
                    runner,
                    duplicate_request,
                ),
                timeout=0.1,
            )

        assert await runner_cls.cancel_query_output_block_lease_request(
            runner,
            first_request["request_id"],
        ) == {"cancelled": True, "released": False}
        denial = await first_waiter
        assert denial["blocked_reason"] == "output_lease_request_cancelled"
        await _release_owned_outputs(runner_cls, runner, owned)
        assert await runner_cls.release_query_task_lease(
            runner,
            task_request["request_id"],
            task["lease"]["lease_id"],
            task["lease"]["attempt_id"],
        ) == {"released": True}

    asyncio.run(scenario())


def test_terminal_task_and_output_requests_are_idempotent_for_query_lifetime():
    async def scenario():
        query_id = "query-terminal-request-replay"
        graph = _graph(query_id)
        runner_cls, runner = _runner(asyncio.get_running_loop())
        manager = register_query_graph(
            graph,
            _allocation(),
            on_change=lambda: runner_cls._signal_query_resource_change(
                runner,
                query_id,
            ),
        )
        manager.update_stage_state(graph.stages[0].stage_id, runnable=True)

        task_request = _task_request(
            query_id,
            "request:terminal-task",
            "task:terminal",
        )
        task = await runner_cls.acquire_query_task_lease(runner, task_request)
        output_request = {
            "request_id": "request:terminal-output",
            "query_id": query_id,
            "producer_stage_id": graph.stages[0].stage_id,
            "task_lease_id": task["lease"]["lease_id"],
            "attempt_id": task["lease"]["attempt_id"],
            "block_id": "block:terminal",
            "size_bytes": 10,
        }
        output = await runner_cls.acquire_query_output_block_lease(
            runner,
            output_request,
        )

        assert await runner_cls.release_query_output_block_lease(
            runner,
            output_request["request_id"],
            output["lease"]["lease_id"],
        ) == {"released": True}
        output_replay = await runner_cls.acquire_query_output_block_lease(
            runner,
            output_request,
        )
        assert output_replay["granted"] is False
        assert output_replay["fatal"] is True
        assert output_replay["blocked_reason"] == "output_lease_request_released"
        assert manager.snapshot()["output_leases"] == {}

        second_output_owner = dict(output_request)
        second_output_owner["request_id"] = "request:second-terminal-output"
        second_output = await runner_cls.acquire_query_output_block_lease(
            runner,
            second_output_owner,
        )
        assert second_output["granted"] is False
        assert second_output["fatal"] is True
        assert second_output["blocked_reason"] == "output_block_terminal"

        assert await runner_cls.release_query_task_lease(
            runner,
            task_request["request_id"],
            task["lease"]["lease_id"],
            task["lease"]["attempt_id"],
        ) == {"released": True}
        task_replay = await runner_cls.acquire_query_task_lease(
            runner,
            task_request,
        )
        assert task_replay["granted"] is False
        assert task_replay["fatal"] is True
        assert task_replay["blocked_reason"] == "task_lease_request_released"
        assert manager.snapshot()["task_leases"] == {}

        second_task_owner = dict(task_request)
        second_task_owner["request_id"] = "request:second-terminal-task"
        second_task = await runner_cls.acquire_query_task_lease(
            runner,
            second_task_owner,
        )
        assert second_task["granted"] is False
        assert second_task["fatal"] is True
        assert second_task["blocked_reason"] == "attempt_terminal"

    asyncio.run(scenario())


def test_cancelled_admission_request_replay_returns_the_same_terminal_denial():
    async def scenario():
        query_id = "query-cancelled-request-replay"
        graph = _graph(query_id)
        runner_cls, runner = _runner(asyncio.get_running_loop())
        manager = register_query_graph(
            graph,
            _allocation(),
            on_change=lambda: runner_cls._signal_query_resource_change(
                runner,
                query_id,
            ),
        )
        manager.update_stage_state(graph.stages[0].stage_id, runnable=True)

        active_request = _task_request(
            query_id,
            "request:active-for-cancel",
            "task:active-for-cancel",
        )
        active = await runner_cls.acquire_query_task_lease(runner, active_request)
        cancelled_request = _task_request(
            query_id,
            "request:cancelled",
            "task:cancelled",
        )
        original_waiter = asyncio.create_task(runner_cls.acquire_query_task_lease(runner, cancelled_request))
        for _ in range(4):
            await asyncio.sleep(0)
        assert not original_waiter.done()

        assert await runner_cls.cancel_query_task_lease_request(
            runner,
            cancelled_request["request_id"],
            submitted=False,
        ) == {"cancelled": True, "released": False}
        original_denial = await original_waiter
        replay_denial = await runner_cls.acquire_query_task_lease(
            runner,
            cancelled_request,
        )
        assert replay_denial == original_denial
        assert replay_denial["blocked_reason"] == "task_lease_request_cancelled"

        second_owner = dict(cancelled_request)
        second_owner["request_id"] = "request:second-cancelled-owner"
        second_denial = await runner_cls.acquire_query_task_lease(runner, second_owner)
        assert second_denial["granted"] is False
        assert second_denial["fatal"] is True
        assert second_denial["blocked_reason"] == "attempt_terminal"

        assert await runner_cls.release_query_task_lease(
            runner,
            active_request["request_id"],
            active["lease"]["lease_id"],
            active["lease"]["attempt_id"],
        ) == {"released": True}

    asyncio.run(scenario())


def test_query_teardown_resolves_all_pending_admission_futures():
    class _CoordinatorStub:
        def release_query(self, query_id, generation):
            return True

        def snapshot(self):
            return {"queries": {}}

    async def scenario():
        query_id = "query-admission-teardown"
        graph = _graph(query_id)
        allocation = _allocation()
        runner_cls, runner = _runner(asyncio.get_running_loop())
        manager = register_query_graph(
            graph,
            allocation,
            on_change=lambda: runner_cls._signal_query_resource_change(runner, query_id),
        )
        stage_id = graph.stages[0].stage_id
        manager.update_stage_state(stage_id, runnable=True)
        runner._query_resource_lock = threading.RLock()
        runner._query_resource_coordinator = _CoordinatorStub()
        runner._query_graphs = {query_id: graph}
        runner._query_allocations = {query_id: allocation}

        active_request = _task_request(query_id, "request:active", "task:active")
        active = await runner_cls.acquire_query_task_lease(runner, active_request)
        await _fill_task_output_window(
            runner_cls,
            runner,
            query_id,
            stage_id,
            active,
            prefix="teardown",
        )
        pending_task_request = _task_request(
            query_id,
            "request:pending",
            "task:pending",
        )
        pending_task = asyncio.create_task(runner_cls.acquire_query_task_lease(runner, pending_task_request))
        pending_output = asyncio.create_task(
            runner_cls.acquire_query_output_block_lease(
                runner,
                {
                    "request_id": "request:output",
                    "query_id": query_id,
                    "producer_stage_id": stage_id,
                    "task_lease_id": active["lease"]["lease_id"],
                    "attempt_id": active["lease"]["attempt_id"],
                    "block_id": "blocked-output",
                    "size_bytes": 10,
                },
            )
        )
        for _ in range(8):
            await asyncio.sleep(0)
        assert not pending_task.done()
        assert not pending_output.done()

        runner_cls._release_query_resources(
            runner,
            query_id,
            reason="test_teardown",
        )
        task_denial, output_denial = await asyncio.wait_for(
            asyncio.gather(pending_task, pending_output),
            timeout=1,
        )
        assert task_denial["blocked_reason"] == "query_not_registered"
        assert task_denial["fatal"] is True
        assert output_denial["blocked_reason"] == "query_not_registered"
        assert output_denial["fatal"] is True
        assert runner._query_task_lease_requests == {}
        assert runner._query_output_lease_requests == {}

    asyncio.run(scenario())


def test_query_teardown_cleans_local_state_when_coordinator_lease_expired():
    from vane.runners.ray.query_resource_runtime import (
        get_query_resource_manager,
    )

    class _ExpiredCoordinator:
        def __init__(self):
            self.calls = []

        def release_query(self, query_id, generation):
            self.calls.append((query_id, generation))
            return False

    async def scenario():
        query_id = "query-expired-coordinator-cleanup"
        graph = _graph(query_id)
        allocation = _allocation()
        runner_cls, runner = _runner(asyncio.get_running_loop())
        runner_cls._ensure_query_resource_admission_state(runner)
        register_query_graph(graph, allocation)
        coordinator = _ExpiredCoordinator()
        runner._query_resource_lock = threading.RLock()
        runner._query_resource_coordinator = coordinator
        runner._query_graphs = {query_id: graph}
        runner._query_allocations = {query_id: allocation}
        runner._synchronize_query_allocations = lambda: None

        with pytest.raises(
            RuntimeError,
            match="coordinator allocation was already absent",
        ):
            runner_cls._release_query_resources(
                runner,
                query_id,
                reason="expired_coordinator",
            )

        with pytest.raises(KeyError):
            get_query_resource_manager(query_id)
        assert runner._query_graphs == {}
        assert runner._query_allocations == {}
        assert coordinator.calls == [(query_id, allocation.generation)]

        # Cleanup is convergent: a retry sees no local owner and succeeds.
        runner_cls._release_query_resources(
            runner,
            query_id,
            reason="cleanup_retry",
        )

    asyncio.run(scenario())


def test_fragment_drop_waits_for_fte_admission_pump_before_registry_drop(monkeypatch):
    import vane.runners.ray.fte_fragment_scheduler as fte_scheduler

    class _CoordinatorStub:
        def release_query(self, query_id, generation):
            return True

        def snapshot(self):
            return {"queries": {}}

    async def scenario():
        query_id = "query-fragment-drop-fence"
        graph = _graph(query_id)
        allocation = _allocation()
        runner_cls, runner = _runner(asyncio.get_running_loop())
        runner_cls._ensure_query_resource_admission_state(runner)
        register_query_graph(graph, allocation)
        runner._query_resource_lock = threading.RLock()
        runner._query_resource_coordinator = _CoordinatorStub()
        runner._query_graphs = {query_id: graph}
        runner._query_allocations = {query_id: allocation}
        runner._synchronize_query_allocations = lambda: None

        drain_entered = threading.Event()
        release_drain = threading.Event()
        fragment_drop_called = threading.Event()
        registry_state = {"live": False}

        def blocked_drain(actual_query_id):
            assert actual_query_id == query_id
            drain_entered.set()
            assert release_drain.wait(2.0)
            # This models the real drain writing handles/watchers/schedulers
            # after taking its execution snapshot.
            registry_state["live"] = True

        monkeypatch.setattr(
            fte_scheduler,
            "drain_fte_resource_admission_change",
            blocked_drain,
        )

        class _PlanRunner:
            def drop_query_fragments(self, actual_query_id):
                assert actual_query_id == query_id
                registry_state["live"] = False
                fragment_drop_called.set()

        runner._get_plan_runner = lambda: _PlanRunner()
        runner._query_fte_admission_dirty_queries.add(query_id)
        done = threading.Event()
        task = asyncio.create_task(runner_cls._run_query_fte_admission_pump(runner, query_id, done))
        runner._query_fte_admission_pumps[query_id] = task
        runner._query_fte_admission_done_events[query_id] = done
        assert await asyncio.to_thread(drain_entered.wait, 1.0)

        teardown = asyncio.create_task(
            asyncio.to_thread(
                runner_cls._drop_query_fragments_sync,
                runner,
                query_id,
            )
        )
        await asyncio.sleep(0.05)
        assert fragment_drop_called.is_set() is False

        release_drain.set()
        await asyncio.wait_for(teardown, timeout=2.0)
        await asyncio.wait_for(task, timeout=1.0)

        assert fragment_drop_called.is_set() is True
        assert registry_state["live"] is False

    asyncio.run(scenario())


def test_fragment_drop_keeps_query_resources_when_local_fte_registry_cannot_quiesce(monkeypatch):
    import vane.runners.ray.fte_fragment_scheduler as fte_scheduler
    from vane.runners.ray.query_resource_runtime import get_query_resource_manager

    class _CoordinatorStub:
        def __init__(self):
            self.released = []

        def release_query(self, query_id, generation):
            self.released.append((query_id, generation))
            return True

    async def scenario():
        query_id = "query-local-fte-quiesce-failure"
        graph = _graph(query_id)
        allocation = _allocation()
        runner_cls, runner = _runner(asyncio.get_running_loop())
        runner_cls._ensure_query_resource_admission_state(runner)
        manager = register_query_graph(graph, allocation)
        coordinator = _CoordinatorStub()
        runner._query_resource_lock = threading.RLock()
        runner._query_resource_coordinator = coordinator
        runner._query_graphs = {query_id: graph}
        runner._query_allocations = {query_id: allocation}
        runner._synchronize_query_allocations = lambda: None
        runner._get_plan_runner = lambda: SimpleNamespace(
            drop_query_fragments=lambda _query_id: None,
        )

        monkeypatch.setattr(
            fte_scheduler,
            "_drop_fte_registry_for_query",
            lambda _query_id: (_ for _ in ()).throw(RuntimeError("watcher still alive")),
        )

        with pytest.raises(RuntimeError, match="local FTE registry did not quiesce"):
            await asyncio.to_thread(
                runner_cls._drop_query_fragments_sync,
                runner,
                query_id,
            )

        assert get_query_resource_manager(query_id) is manager
        assert runner._query_graphs == {query_id: graph}
        assert runner._query_allocations == {query_id: allocation}
        assert coordinator.released == []
        assert runner._query_resource_admission_bridge_poisoned is True

    asyncio.run(scenario())


def test_fragment_drop_retains_query_owner_while_remote_teardown_is_incomplete(monkeypatch):
    import vane.runners.ray.fte_fragment_scheduler as fte_scheduler
    from vane.runners.ray.driver import QueryTeardownOwnershipError
    from vane.runners.ray.query_resource_runtime import get_query_resource_manager

    class _CoordinatorStub:
        def __init__(self):
            self.released = []

        def release_query(self, query_id, generation):
            self.released.append((query_id, generation))
            return True

    async def scenario():
        query_id = "query-remote-teardown-owner"
        graph = _graph(query_id)
        allocation = _allocation()
        runner_cls, runner = _runner(asyncio.get_running_loop())
        runner_cls._ensure_query_resource_admission_state(runner)
        manager = register_query_graph(graph, allocation)
        coordinator = _CoordinatorStub()
        runner._query_resource_lock = threading.RLock()
        runner._query_resource_coordinator = coordinator
        runner._query_graphs = {query_id: graph}
        runner._query_allocations = {query_id: allocation}
        runner._synchronize_query_allocations = lambda: None
        runner._fence_query_resource_admission_for_teardown = lambda _query_id: None

        class _PlanRunner:
            @staticmethod
            def drop_query_fragments(_query_id):
                raise TimeoutError("planned worker teardown timeout")

        runner._get_plan_runner = lambda: _PlanRunner()
        monkeypatch.setattr(
            fte_scheduler,
            "fte_query_remote_teardown_blockers",
            lambda actual_query_id: (
                (
                    "active_teardown=1",
                    "worker_teardown=worker-0",
                )
                if actual_query_id == query_id
                else ()
            ),
        )

        with pytest.raises(
            QueryTeardownOwnershipError,
            match="retains remote ownership",
        ) as exc_info:
            await asyncio.to_thread(
                runner_cls._drop_query_fragments_sync,
                runner,
                query_id,
            )

        assert "planned worker teardown timeout" in str(exc_info.value)
        assert "active_teardown=1" in str(exc_info.value)
        assert get_query_resource_manager(query_id) is manager
        assert runner._query_graphs == {query_id: graph}
        assert runner._query_allocations == {query_id: allocation}
        assert coordinator.released == []

    asyncio.run(scenario())


def test_owner_loop_sync_fence_times_out_and_cancels_late_callback():
    from vane.runners.ray.driver import RayQueryDriverActor

    class _StalledLoop:
        def __init__(self):
            self.callbacks = []

        @staticmethod
        def is_closed():
            return False

        @staticmethod
        def is_running():
            return True

        def call_soon_threadsafe(self, callback):
            self.callbacks.append(callback)

    runner_cls = RayQueryDriverActor.__ray_metadata__.modified_class
    runner = object.__new__(runner_cls)
    loop = _StalledLoop()
    runner._query_resource_admission_loop = loop
    mutations = []

    with pytest.raises(
        RuntimeError,
        match="timed out waiting for query admission owner-loop fence",
    ):
        runner_cls._run_on_query_resource_admission_loop_sync(
            runner,
            lambda: mutations.append("late"),
            timeout_s=0.01,
        )

    assert mutations == []
    assert len(loop.callbacks) == 1
    loop.callbacks[0]()
    assert mutations == []


def test_owner_loop_sync_fence_poisoned_after_started_callback_timeout():
    from vane.runners.ray.driver import RayQueryDriverActor

    class _RunningLoop:
        def __init__(self):
            self.thread = None

        @staticmethod
        def is_closed():
            return False

        @staticmethod
        def is_running():
            return True

        def call_soon_threadsafe(self, callback):
            self.thread = threading.Thread(target=callback)
            self.thread.start()

    runner_cls = RayQueryDriverActor.__ray_metadata__.modified_class
    runner = object.__new__(runner_cls)
    loop = _RunningLoop()
    runner._query_resource_admission_loop = loop
    callback_started = threading.Event()
    release_callback = threading.Event()
    mutations = []

    def blocked_callback():
        callback_started.set()
        release_callback.wait(timeout=1)
        mutations.append("late")

    with pytest.raises(RuntimeError, match="callback started but did not finish; bridge poisoned"):
        runner_cls._run_on_query_resource_admission_loop_sync(
            runner,
            blocked_callback,
            timeout_s=0.01,
        )
    assert callback_started.is_set()
    assert runner._query_resource_admission_bridge_poisoned is True

    release_callback.set()
    loop.thread.join(timeout=1)
    assert mutations == ["late"]
    with pytest.raises(RuntimeError, match="fence is poisoned"):
        runner_cls._run_on_query_resource_admission_loop_sync(
            runner,
            lambda: None,
            timeout_s=0.01,
        )


def test_query_registration_open_failure_rolls_back_every_owner(monkeypatch):
    import vane.runners.ray.query_graph_builder as graph_builder
    from vane.runners.ray.query_resource_runtime import (
        get_query_resource_manager,
    )

    query_id = "query-registration-open-failure"
    graph = _graph(query_id)
    allocation = _allocation()
    released: list[tuple[str, int]] = []

    class _Coordinator:
        def register_query(self, demand):
            assert demand == "demand"
            return allocation

        def release_query(self, released_query_id, generation):
            released.append((released_query_id, generation))
            return True

    monkeypatch.setattr(
        graph_builder,
        "build_query_execution_graph",
        lambda _metadata: graph,
    )
    monkeypatch.setattr(
        graph_builder,
        "build_query_demand",
        lambda _graph, _capacity: "demand",
    )

    from vane.runners.ray.driver import RayQueryDriverActor

    runner_cls = RayQueryDriverActor.__ray_metadata__.modified_class
    runner = object.__new__(runner_cls)
    runner._duckdb_conn = None
    runner._query_resource_lock = threading.RLock()
    runner._query_graphs = {}
    runner._query_allocations = {}
    runner._query_resource_coordinator = _Coordinator()
    runner._refresh_query_capacity = lambda: ResourceVector(cpu=1, heap_bytes=101, object_store_bytes=20)
    runner._synchronize_query_allocations = lambda: None

    def fail_open(_callback):
        raise RuntimeError("injected owner-loop open failure")

    runner._run_on_query_resource_admission_loop_sync = fail_open
    plan = SimpleNamespace(
        collect_execution_stages=lambda conn: object(),
        idx=lambda: query_id,
    )

    with pytest.raises(RuntimeError, match="injected owner-loop open failure"):
        runner_cls._register_query_resources(runner, plan)

    with pytest.raises(KeyError):
        get_query_resource_manager(query_id)
    assert released == [(query_id, allocation.generation)]
    assert runner._query_graphs == {}
    assert runner._query_allocations == {}


def test_background_query_teardown_fences_new_admission_before_table_purge():
    class _CoordinatorStub:
        def release_query(self, query_id, generation):
            return True

        def snapshot(self):
            return {"queries": {}}

    async def scenario():
        query_id = "query-background-teardown-fence"
        graph = _graph(query_id)
        allocation = _allocation()
        runner_cls, runner = _runner(asyncio.get_running_loop())
        manager = register_query_graph(graph, allocation)
        manager.update_stage_state(graph.stages[0].stage_id, runnable=True)
        runner._query_resource_lock = threading.RLock()
        runner._query_resource_coordinator = _CoordinatorStub()
        runner._query_graphs = {query_id: graph}
        runner._query_allocations = {query_id: allocation}

        active_request = _task_request(
            query_id,
            "request:active",
            "task:active",
        )
        active = await runner_cls.acquire_query_task_lease(runner, active_request)
        assert active["granted"]

        owner_thread_id = threading.get_ident()
        failed_pending = threading.Event()
        allow_background_teardown = threading.Event()
        original_fail = runner_cls._fail_query_admission_requests.__get__(
            runner,
            runner_cls,
        )

        def controlled_fail(query_key):
            original_fail(query_key)
            failed_pending.set()
            if threading.get_ident() != owner_thread_id:
                assert allow_background_teardown.wait(timeout=1)

        runner._fail_query_admission_requests = controlled_fail
        teardown = asyncio.create_task(
            asyncio.to_thread(
                runner_cls._release_query_resources,
                runner,
                query_id,
                reason="test_background_teardown",
            )
        )
        await asyncio.to_thread(failed_pending.wait, 1)

        late_request = _task_request(
            query_id,
            "request:late",
            "task:late",
        )
        late_waiter = asyncio.create_task(runner_cls.acquire_query_task_lease(runner, late_request))
        for _ in range(4):
            await asyncio.sleep(0)
        allow_background_teardown.set()
        await teardown

        denial = await asyncio.wait_for(late_waiter, timeout=0.2)
        assert denial["granted"] is False
        assert denial["fatal"] is True
        assert denial["blocked_reason"] == "query_not_registered"
        assert runner._query_task_lease_requests == {}

    asyncio.run(scenario())


def test_task_admission_pump_resolves_waiter_when_stage_becomes_terminal():
    async def scenario():
        query_id = "query-task-stage-terminal"
        graph = _graph(query_id)
        runner_cls, runner = _runner(asyncio.get_running_loop())
        manager = register_query_graph(
            graph,
            _allocation(),
            on_change=lambda: runner_cls._signal_query_resource_change(runner, query_id),
        )
        stage_id = graph.stages[0].stage_id
        manager.update_stage_state(stage_id, runnable=True)
        active_request = _task_request(query_id, "request:active", "task:active")
        active = await runner_cls.acquire_query_task_lease(runner, active_request)
        pending_request = _task_request(
            query_id,
            "request:pending",
            "task:pending",
        )
        pending = asyncio.create_task(
            runner_cls.acquire_query_task_lease(
                runner,
                pending_request,
            )
        )
        for _ in range(5):
            await asyncio.sleep(0)
        assert not pending.done()

        manager.update_stage_state(stage_id, runnable=False, completed=True)
        denial = await asyncio.wait_for(pending, timeout=1)
        assert denial["granted"] is False
        assert denial["fatal"] is True
        assert denial["blocked_reason"] == "stage_completed"
        assert (
            await runner_cls.acquire_query_task_lease(
                runner,
                pending_request,
            )
            == denial
        )
        assert await runner_cls.release_query_task_lease(
            runner,
            active_request["request_id"],
            active["lease"]["lease_id"],
            active["lease"]["attempt_id"],
        ) == {"released": True}

    asyncio.run(scenario())


def test_query_resource_signals_are_coalesced_before_event_loop_dispatch():
    async def scenario():
        runner_cls, runner = _runner(asyncio.get_running_loop())
        task_pumps = []
        output_pumps = []
        runner._schedule_query_task_admission_pump = task_pumps.append
        runner._schedule_query_output_admission_pump = output_pumps.append

        for _ in range(20):
            runner_cls._signal_query_resource_change(runner, "query-coalesced")
        for _ in range(3):
            await asyncio.sleep(0)

        assert task_pumps == ["query-coalesced"]
        assert output_pumps == ["query-coalesced"]

    asyncio.run(scenario())


def test_new_query_generation_reopens_admission_after_old_state_is_purged():
    async def scenario():
        query_id = "query-generation-reopen"
        graph = _graph(query_id)
        runner_cls, runner = _runner(asyncio.get_running_loop())
        register_query_graph(graph, _allocation())

        runner_cls._close_query_resource_admission(runner, query_id)
        assert query_id in runner._query_resource_closing_queries
        clear_query_resource_managers()

        manager = register_query_graph(graph, _allocation())
        manager.update_stage_state(graph.stages[0].stage_id, runnable=True)
        runner_cls._open_query_resource_admission(runner, query_id)
        grant = await runner_cls.acquire_query_task_lease(
            runner,
            _task_request(query_id, "request:new-generation", "task:new-generation"),
        )

        assert grant["granted"] is True
        assert query_id not in runner._query_resource_closing_queries

    asyncio.run(scenario())
