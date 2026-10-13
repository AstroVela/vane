# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import threading
from concurrent.futures import Future
from types import SimpleNamespace

import pyarrow as pa
import pytest


def _payload(**overrides):
    value = {
        "execution_backend": "ray_actor",
        "query_id": "q1",
        "resource_unit_id": "resource:q1:udf:node:1",
        "produce_ray_block_stream": True,
        "actor_pool_size": 1,
        "cpus": 1.0,
        "gpus": 1.0,
        "memory_bytes": 1024,
        "udf_task_input_max_bytes": 1024,
        "udf_output_target_max_bytes": 1024,
        "output_window_bytes": 2048,
        "call_mode": "map_batches",
        "output_schema": {"value": "BIGINT"},
    }
    value.update(overrides)
    return value


class _Method:
    def __init__(self, *, result="generator", error=None):
        self.options_calls = []
        self.remote_calls = []
        self.result = result
        self.error = error

    def options(self, **kwargs):
        self.options_calls.append(kwargs)
        return self

    def remote(self, *args, **kwargs):
        self.remote_calls.append((args, kwargs))
        if self.error is not None:
            raise self.error
        return self.result


class _Actor:
    def __init__(self, **method_kwargs):
        self.run_block_stream = _Method(**method_kwargs)
        self.run_ref_bundle_stream = _Method(**method_kwargs)


class _InitRef:
    def __init__(self, future):
        self._future = future

    def future(self):
        return self._future


def _executor(actors, *, dispatch_indices=None, payload=None, node_ids=None, init_refs=None):
    from vane.execution.udf_ray import RemoteUDFExecutor, UDFActorPool

    pool = UDFActorPool._from_handles(
        actors,
        payload=payload or _payload(),
        actor_node_ids=node_ids or ["node-a"] * len(actors),
        actor_dispatch_indices=(set(range(len(actors))) if dispatch_indices is None else dispatch_indices),
        actor_init_refs=init_refs,
    )
    return RemoteUDFExecutor(
        pool,
        payload or _payload(),
        query_driver_handle=object(),
        query_generation_capability="test-query-generation-capability",
    )


def _run_after_lease(executor, monkeypatch, *, actor_index=0):
    import vane.execution.udf_ray_remote_ref_bundle as remote_ref_bundle
    import vane.execution.udf_ray_remote_submit as remote_submit

    lease = {
        "query_id": executor._payload["query_id"],
        "resource_unit_id": executor._payload["resource_unit_id"],
        "lease_id": f"lease-{actor_index}",
        "attempt_id": f"attempt-{actor_index}",
        "node_id": "node-a",
        "execution_slot_id": f"ray_actor:{executor._payload['resource_unit_id']}:{actor_index}",
        "actor_index": actor_index,
        "output_window_bytes": executor._payload["output_window_bytes"],
    }
    admission = SimpleNamespace(
        driver=object(),
        request_id=f"request-{actor_index}",
        lease=lease,
    )
    monkeypatch.setattr(executor, "_take_task_admission", lambda: admission)

    def _start_immediately(*, admission, submitter, **_kwargs):
        return submitter(dict(admission.lease))

    monkeypatch.setattr(remote_submit, "TaskLeaseObjectRefGenerator", _start_immediately)
    monkeypatch.setattr(remote_ref_bundle, "TaskLeaseObjectRefGenerator", _start_immediately)


def _submit_kind(executor, kind):
    if kind == "table":
        return executor.submit_with_id(1, pa.table({"value": [42]}))
    return executor.submit_ref_bundle_with_id(1, [object()], [None], [{"num_rows": 1, "size_bytes": 8}], ["value"])


def test_actor_dispatch_uses_only_dispatch_eligible_actor(monkeypatch):
    actors = [_Actor(), _Actor()]
    executor = _executor(actors, dispatch_indices={1})
    _run_after_lease(executor, monkeypatch, actor_index=1)

    result = executor.submit_with_id(1, pa.table({"value": [1]}))

    assert result == "generator"
    assert actors[0].run_block_stream.remote_calls == []
    assert len(actors[1].run_block_stream.remote_calls) == 1
    assert actors[1].run_block_stream.options_calls == [{"_generator_backpressure_num_objects": 4}]
    task_payload = actors[1].run_block_stream.remote_calls[0][1]["payload"]
    assert task_payload["task_lease_id"] == "lease-1"


def test_row_preserving_actor_submit_renames_only_udf_argument_prefix(monkeypatch):
    actor = _Actor()
    payload = _payload(
        call_mode="map_batches_rows",
        input_names=["text"],
        scalar_arg_count=1,
        row_preserving=True,
    )
    executor = _executor([actor], payload=payload)
    _run_after_lease(executor, monkeypatch)
    layout = pa.table(
        {
            "physical_arg": ["alpha", "beta"],
            "keep_id": [1, 2],
            "keep_text": ["alpha", "beta"],
        }
    )

    result = executor.submit_with_id(2, layout)

    assert result == "generator"
    submitted = actor.run_block_stream.remote_calls[0][0][0]
    assert submitted.column_names == ["text", "keep_id", "keep_text"]
    assert submitted.to_pydict() == {
        "text": ["alpha", "beta"],
        "keep_id": [1, 2],
        "keep_text": ["alpha", "beta"],
    }


def test_row_preserving_actor_submit_validates_input_names_against_arg_count():
    payload = _payload(
        call_mode="map_batches_rows",
        input_names=["left", "right"],
        scalar_arg_count=1,
        row_preserving=True,
    )
    executor = _executor([_Actor()], payload=payload)

    with pytest.raises(ValueError, match=r"input_names count 2 does not match scalar_arg_count 1"):
        executor.submit_with_id(3, pa.table({"arg": [1], "keep": [2]}))


def test_no_dispatch_eligible_actor_is_a_hard_error_after_admission(monkeypatch):
    actor = _Actor()
    executor = _executor([actor], dispatch_indices=set())
    _run_after_lease(executor, monkeypatch)

    with pytest.raises(RuntimeError, match="actor_index=0, but that actor is not ready"):
        executor.submit_with_id(2, pa.table({"value": [1]}))
    assert actor.run_block_stream.remote_calls == []


@pytest.mark.parametrize("kind", ["table", "ref_bundle"])
@pytest.mark.parametrize("actor_index", [0, 1])
def test_first_actor_dispatch_precedes_executor_ready_notification(monkeypatch, kind, actor_index):
    actors = [_Actor(), _Actor()]
    futures = [Future(), Future()]
    callback_entered, callback_release = threading.Event(), threading.Event()

    def delay_notification(_future):
        callback_entered.set()
        assert callback_release.wait(10)

    # A completed Future can already grant admission in the driver while an
    # earlier callback keeps the executor's notification from running.
    futures[actor_index].add_done_callback(delay_notification)
    executor = _executor(
        actors,
        dispatch_indices=set(),
        payload=_payload(actor_pool_size=2),
        node_ids=["", ""],
        init_refs=[_InitRef(future) for future in futures],
    )
    _run_after_lease(executor, monkeypatch, actor_index=actor_index)
    completion = threading.Thread(target=lambda: futures[actor_index].set_result("node-a"))
    completion.start()
    try:
        assert callback_entered.wait(5)
        assert not executor._ready_actor_set and not executor._ready_probe_refs
        assert futures[actor_index].done()
        assert _submit_kind(executor, kind) == "generator"
        selected = getattr(actors[actor_index], "run_block_stream" if kind == "table" else "run_ref_bundle_stream")
        assert len(selected.remote_calls) == 1
        assert selected.remote_calls[0][1]["payload"]["actor_index"] == actor_index
        assert not actors[1 - actor_index].run_block_stream.remote_calls
        assert not actors[1 - actor_index].run_ref_bundle_stream.remote_calls
        assert executor._ready_actor_set == {actor_index}
        assert executor._actors_obj.actor_node_ids[actor_index] == "node-a"
        callback_release.set()
        completion.join(5)
        assert not completion.is_alive()
        executor._refresh_actor_readiness()
        assert executor._ready_actor_indices == [actor_index]
        pending_ref = executor._actors_obj._init_refs[1 - actor_index]
        assert executor._pending_ready_refs == {pending_ref: 1 - actor_index}
    finally:
        callback_release.set()
        completion.join(5)
        executor.shutdown()


@pytest.mark.parametrize("kind", ["table", "ref_bundle"])
def test_admission_waits_for_local_init_result_delivery(monkeypatch, kind):
    resolving = threading.Event()

    class ObservedFuture(Future):
        def result(self, timeout=None):
            resolving.set()
            return super().result(timeout=timeout)

    actors = [_Actor(), _Actor()]
    future = ObservedFuture()
    executor = _executor(
        actors,
        dispatch_indices=set(),
        payload=_payload(actor_pool_size=2),
        node_ids=["", ""],
        init_refs=[_InitRef(Future()), _InitRef(future)],
    )
    _run_after_lease(executor, monkeypatch, actor_index=1)
    monkeypatch.setenv("VANE_RAY_OBJECT_GET_TIMEOUT_S", "5")

    def deliver_result():
        assert resolving.wait(5)
        future.set_result("node-a")

    # Driver admission has already confirmed initialization. The same result
    # reaches the executor only once submission starts resolving its local ref.
    completion = threading.Thread(target=deliver_result)
    completion.start()
    try:
        assert not future.done()
        assert _submit_kind(executor, kind) == "generator"
        assert executor._ready_actor_set == {1}
        selected = getattr(actors[1], "run_block_stream" if kind == "table" else "run_ref_bundle_stream")
        assert len(selected.remote_calls) == 1
        assert not actors[0].run_block_stream.remote_calls
        assert not actors[0].run_ref_bundle_stream.remote_calls
    finally:
        resolving.set()
        completion.join(5)
        executor.shutdown()


@pytest.mark.parametrize("kind", ["table", "ref_bundle"])
@pytest.mark.parametrize("outcome", ["pending", "failed", "empty_node"])
def test_admitted_actor_still_requires_successful_initialization(monkeypatch, kind, outcome):
    actor, future = _Actor(), Future()
    executor = _executor([actor], dispatch_indices=set(), node_ids=[""], init_refs=[_InitRef(future)])
    _run_after_lease(executor, monkeypatch)
    monkeypatch.setattr(executor, "_notify_ready_ref_ready", lambda _ref: None)
    if outcome == "failed":
        future.set_exception(ValueError("model initialization failed"))
        error_type, message = ValueError, "model initialization failed"
    elif outcome == "empty_node":
        future.set_result("")
        error_type, message = RuntimeError, "without a runtime node identity"
    else:
        monkeypatch.setenv("VANE_RAY_OBJECT_GET_TIMEOUT_S", "0.01")
        error_type, message = TimeoutError, ""
    try:
        with pytest.raises(error_type, match=message):
            _submit_kind(executor, kind)
        assert not actor.run_block_stream.remote_calls
        assert not actor.run_ref_bundle_stream.remote_calls
        assert not executor._ready_actor_set
    finally:
        executor.shutdown()


def test_actor_submit_failure_marks_actor_unavailable_without_cross_actor_retry(monkeypatch):
    failed = _Actor(error=RuntimeError("actor died"))
    healthy = _Actor()
    executor = _executor([failed, healthy], dispatch_indices={0, 1})
    _run_after_lease(executor, monkeypatch)

    with pytest.raises(RuntimeError, match="actor_idx=0"):
        executor.submit_with_id(3, pa.table({"value": [1]}))

    assert len(failed.run_block_stream.remote_calls) == 1
    assert healthy.run_block_stream.remote_calls == []
    assert 0 not in executor._ready_actor_set


def test_ref_bundle_dispatch_uses_direct_input_refs_and_block_stream(monkeypatch):
    actor = _Actor()
    executor = _executor([actor])
    _run_after_lease(executor, monkeypatch)
    block = object()

    result = executor.submit_ref_bundle_with_id(
        4,
        [block],
        [None],
        [{"num_rows": 1, "size_bytes": 64}],
        ["value"],
    )

    assert result == "generator"
    args, kwargs = actor.run_ref_bundle_stream.remote_calls[0]
    assert args == (block,)
    assert kwargs["metadata"] == [{"num_rows": 1, "size_bytes": 64}]
    assert kwargs["payload"]["task_lease_id"] == "lease-0"
    assert kwargs["payload"]["actor_index"] == 0


def test_existing_actor_handles_require_explicit_dispatch_eligibility():
    from vane.execution.udf_ray import UDFActorPool

    actor = _Actor()
    pool = UDFActorPool._from_handles(
        [actor],
        payload=_payload(),
        actor_node_ids=["node-a"],
        actor_dispatch_indices=set(),
    )
    executor = __import__("vane.execution.udf_ray", fromlist=["RemoteUDFExecutor"]).RemoteUDFExecutor(
        pool,
        _payload(),
        query_driver_handle=object(),
        query_generation_capability="test-query-generation-capability",
    )
    assert executor._ready_actor_indices == []
    assert executor._actor_init_errors[0] == "ray actor does not expose __ray_ready__ readiness probe"


def test_pending_actor_learns_runtime_node_and_joins_dispatch_set():
    from concurrent.futures import Future

    from vane.execution.udf_ray import RemoteUDFExecutor, UDFActorPool

    class _Ref:
        def __init__(self, future):
            self._future = future

        def future(self):
            return self._future

    actors = [_Actor(), _Actor()]
    first_future = Future()
    second_future = Future()
    pool = UDFActorPool._from_handles(
        actors,
        payload=_payload(actor_pool_size=2),
        actor_node_ids=["node-a", ""],
        actor_dispatch_indices={0},
        actor_init_refs=[_Ref(first_future), _Ref(second_future)],
    )
    executor = RemoteUDFExecutor(
        pool,
        _payload(actor_pool_size=2),
        query_driver_handle=object(),
        query_generation_capability="test-query-generation-capability",
    )

    second_future.set_result("node-b")
    executor._refresh_actor_readiness()

    assert executor._actor_node_ids == ["node-a", "node-b"]
    assert executor._ready_actor_set == {0, 1}


def test_actor_executor_uses_explicit_query_driver_handle():
    from vane.execution.udf_ray import _build_ray_actor_executor

    actor = _Actor()
    query_driver_handle = object()
    options = {
        "actor_handles": [actor],
        "actor_node_ids": ["node-a"],
        "actor_dispatch_indices": [0],
    }

    with pytest.raises(RuntimeError, match="explicit query driver handle"):
        _build_ray_actor_executor(_payload(), options)

    with pytest.raises(RuntimeError, match="query generation capability"):
        _build_ray_actor_executor(
            _payload(),
            {
                **options,
                "query_driver_handle": query_driver_handle,
            },
        )

    with pytest.raises(RuntimeError, match="explicit Vane session config"):
        _build_ray_actor_executor(
            _payload(),
            {
                **options,
                "query_driver_handle": query_driver_handle,
                "query_generation_capability": "test-query-generation-capability",
            },
        )

    executor = _build_ray_actor_executor(
        _payload(),
        {
            **options,
            "query_driver_handle": query_driver_handle,
            "query_generation_capability": "test-query-generation-capability",
            "session_config": {},
        },
    )
    assert executor._task_admission._driver is query_driver_handle


def test_actor_executor_options_allow_pending_identity_but_require_it_for_ready_slots():
    from vane.execution.udf_ray import _build_udf_executor_options

    actor = _Actor()
    pending = _build_udf_executor_options(
        actor_handles=[actor],
        actor_node_ids=None,
        actor_dispatch_indices=set(),
        actor_init_refs=[object()],
    )
    assert pending["actor_node_ids"] == [""]
    assert pending["actor_dispatch_indices"] == []
    with pytest.raises(ValueError, match="dispatch-ready actors require runtime node IDs"):
        _build_udf_executor_options(
            actor_handles=[actor],
            actor_node_ids=None,
            actor_dispatch_indices={0},
        )
    with pytest.raises(ValueError, match="out-of-range"):
        _build_udf_executor_options(
            actor_handles=[actor],
            actor_node_ids=["node-a"],
            actor_dispatch_indices={1},
        )
    with pytest.raises(ValueError, match="integers"):
        _build_udf_executor_options(
            actor_handles=[actor],
            actor_node_ids=["node-a"],
            actor_dispatch_indices={"0"},
        )


def test_actor_lifecycle_shutdown_is_idempotent_for_borrowed_handles():
    executor = _executor([_Actor()])
    executor.shutdown()
    executor.shutdown()
    assert executor.actors
