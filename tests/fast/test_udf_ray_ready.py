# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import pyarrow as pa
import pytest


def _payload(**overrides):
    value = {
        "execution_backend": "ray_actor",
        "query_id": "q1",
        "stage_id": "stage:q1:node:1:udf",
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


def _executor(actors, *, dispatch_indices=None, payload=None, node_ids=None):
    from vane.execution.udf_ray import RemoteUDFExecutor, UDFActorPool

    pool = UDFActorPool._from_handles(
        actors,
        payload=payload or _payload(),
        actor_node_ids=node_ids or ["node-a"] * len(actors),
        actor_dispatch_indices=(set(range(len(actors))) if dispatch_indices is None else dispatch_indices),
    )
    return RemoteUDFExecutor(pool, payload or _payload())


def _run_after_lease(executor, monkeypatch, *, actor_index=0):
    import vane.execution.udf_ray_remote_ref_bundle as remote_ref_bundle
    import vane.execution.udf_ray_remote_submit as remote_submit

    lease = {
        "query_id": executor._payload["query_id"],
        "stage_id": executor._payload["stage_id"],
        "lease_id": f"lease-{actor_index}",
        "attempt_id": f"attempt-{actor_index}",
        "node_id": "node-a",
        "execution_slot_id": f"ray_actor:{executor._payload['stage_id']}:{actor_index}",
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

    with pytest.raises(RuntimeError, match="no ready actors"):
        executor.submit_with_id(2, pa.table({"value": [1]}))
    assert actor.run_block_stream.remote_calls == []


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
    executor = __import__("vane.execution.udf_ray", fromlist=["RemoteUDFExecutor"]).RemoteUDFExecutor(pool, _payload())
    assert executor._ready_actor_indices == []
    assert executor._actor_init_errors[0] == "ray actor does not expose __ray_ready__ readiness probe"


def test_actor_executor_options_reject_missing_or_invalid_coordinator_identity():
    from vane.execution.udf_ray import _build_udf_executor_options

    actor = _Actor()
    with pytest.raises(ValueError, match="actor node IDs are required"):
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
