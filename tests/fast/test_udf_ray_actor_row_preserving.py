# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Ray actor block streams must split/fuse row-preserving inputs."""

from __future__ import annotations

import os
import sys
import types
from concurrent.futures import Future
from typing import Any

import pyarrow as pa
import pytest


def _pickle_function(fn: Any) -> bytes:
    from vane.pickle import dumps

    return dumps(fn)


def _rows_payload(fn: Any) -> dict[str, Any]:
    return {
        "function_pickle": _pickle_function(fn),
        "call_mode": "map_batches_rows",
        "execution_backend": "ray_actor",
        "input_names": ["x"],
        "output_schema": [
            {
                "name": "y",
                "kind": "duckdb_type",
                "type": "INTEGER",
                "dtype": "VARCHAR",
                "shape": [],
            }
        ],
        "scalar_arg_count": 1,
        "row_preserving": True,
        "prebatched_input": False,
        "actor_number": 1,
        "produce_ray_block_stream": True,
        "query_id": "query-row-preserving",
        "resource_unit_id": "resource:test:udf:row-preserving",
        "task_lease_id": "lease-row-preserving",
        "attempt_id": "attempt-row-preserving",
        "node_id": "node-row-preserving",
        "udf_output_target_max_bytes": 1 << 20,
        "output_window_bytes": 2 << 20,
    }


class _FakeRuntimeContext:
    def __init__(self, ray_module: _FakeRayModule) -> None:
        self._ray_module = ray_module

    def get_node_id(self) -> str:
        return self._ray_module.node_id

    def get_job_id(self) -> str:
        return "job-row-preserving"

    @property
    def was_current_actor_reconstructed(self) -> bool:
        return self._ray_module.reconstructed


class _FakeReportRef:
    def __init__(self, result: dict[str, Any]) -> None:
        self._result = result

    def future(self) -> Future:
        future = Future()
        future.set_result(dict(self._result))
        return future


class _FakeReportMethod:
    def __init__(self, ray_module: _FakeRayModule) -> None:
        self._ray_module = ray_module

    def remote(self, payload: dict[str, Any], capability: str) -> _FakeReportRef:
        self._ray_module.location_reports.append((dict(payload), str(capability)))
        return _FakeReportRef(
            {
                "accepted": True,
                "node_id": payload["node_id"],
                "moved_task_lease_count": 1,
            }
        )


class _FakeDriver:
    def __init__(self, ray_module: _FakeRayModule) -> None:
        self.report_query_udf_actor_location = _FakeReportMethod(ray_module)


class _FakeRayModule(types.ModuleType):
    def __init__(self) -> None:
        super().__init__("ray")
        self.node_id = "node-row-preserving"
        self.reconstructed = False
        self.location_reports: list[tuple[dict[str, Any], str]] = []
        self._runtime_context = _FakeRuntimeContext(self)
        self._driver = _FakeDriver(self)

    def remote(self, *args: Any, **kwargs: Any):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]

        def deco(cls: Any) -> Any:
            return cls

        return deco

    def get_runtime_context(self) -> _FakeRuntimeContext:
        return self._runtime_context

    def get_actor(self, name: str, *, namespace: str) -> _FakeDriver:
        assert name == "vane-query-runtime-job-row-preserving"
        assert namespace == "vane"
        return self._driver


@pytest.fixture()
def fake_ray(monkeypatch: pytest.MonkeyPatch) -> _FakeRayModule:
    # Load the actor runtime and its runner dependencies against real Ray.
    # The fake only needs to replace Ray while the actor class itself runs.
    import vane.execution.udf_ray_actor_runtime  # noqa: F401
    from vane.runners.ray.ray_env import build_session_runtime_env_vars

    module = _FakeRayModule()
    monkeypatch.setitem(sys.modules, "ray", module)
    monkeypatch.setenv("VANE_ISSUE75_INHERITED_SECRET", "inherited-secret")
    for key, value in build_session_runtime_env_vars({"AWS_ISSUE75_ACTOR_SESSION": "actor-session"}).items():
        monkeypatch.setenv(key, value)
    return module


class _AddOne:
    def __call__(self, table: pa.Table) -> pa.Table:
        return pa.table({"y": [value + 1 for value in table.column("x").to_pylist()]})


class _HeterogeneousBatches:
    def __call__(self, table: pa.Table) -> pa.Table:
        value = table.column("x")[0].as_py()
        output = pa.array([b"first"], type=pa.binary()) if value == 1 else pa.array(["second"])
        return pa.table({"y": output})


def _make_actor(payload: dict[str, Any]):
    from vane.execution.udf_ray_actor_runtime import _actor_class

    actor_cls = _actor_class(max_restarts=0, max_task_retries=0)
    actor = actor_cls()
    actor.init_payload(payload)
    return actor


def _data_blocks(stream_items: list[Any]) -> list[pa.Table]:
    assert len(stream_items) % 2 == 0
    blocks = stream_items[::2]
    metadata = stream_items[1::2]
    assert all(isinstance(block, pa.Table) for block in blocks)
    assert [item["num_rows"] for item in metadata] == [block.num_rows for block in blocks]
    return blocks


def test_actor_block_stream_rows_mode_fuses_passthrough(fake_ray):
    actor = _make_actor(_rows_payload(_AddOne))
    layout = pa.table({"x": [1, 2, 3], "keep": ["a", "b", "c"]})

    blocks = _data_blocks(list(actor.run_block_stream(layout)))

    assert pa.concat_tables(blocks).to_pydict() == {
        "keep": ["a", "b", "c"],
        "y": [2, 3, 4],
    }


def test_actor_block_stream_rows_mode_fuses_heterogeneous_output_pieces(fake_ray):
    payload = _rows_payload(_HeterogeneousBatches)
    payload["batch_size"] = 1
    payload["output_schema"][0]["type"] = "VARCHAR"
    actor = _make_actor(payload)
    layout = pa.table({"x": [1, 2], "keep": ["a", "b"]})

    blocks = _data_blocks(list(actor.run_block_stream(layout)))

    assert len(blocks) == 2
    assert [block.column("y").type for block in blocks] == [pa.binary(), pa.string()]
    assert [block.column("keep").to_pylist() for block in blocks] == [["a"], ["b"]]


def test_actor_constructor_installs_only_explicit_session_environment(fake_ray):
    _make_actor(_rows_payload(_AddOne))

    assert "VANE_ISSUE75_INHERITED_SECRET" not in os.environ
    assert os.environ["AWS_ISSUE75_ACTOR_SESSION"] == "actor-session"


def test_actor_ref_bundle_block_stream_rows_mode_fuses_passthrough(fake_ray):
    actor = _make_actor(_rows_payload(_AddOne))
    block = pa.table({"x": [1, 2], "keep": ["a", "b"]})

    blocks = _data_blocks(
        list(
            actor.run_ref_bundle_stream(
                block,
                slices=[(0, 2)],
                metadata=[{"num_rows": 2}],
                names=["x", "keep"],
            )
        )
    )

    assert pa.concat_tables(blocks).to_pydict() == {
        "keep": ["a", "b"],
        "y": [2, 3],
    }


def test_actor_rows_mode_zero_rows_yields_empty_fused_block(fake_ray):
    actor = _make_actor(_rows_payload(_AddOne))
    layout = pa.table(
        {
            "x": pa.array([], type=pa.int64()),
            "keep": pa.array([], type=pa.string()),
        }
    )

    blocks = _data_blocks(list(actor.run_block_stream(layout)))

    assert len(blocks) == 1
    assert blocks[0].num_rows == 0
    assert blocks[0].column_names == ["keep", "y"]


def test_actor_rows_mode_reuses_executor_across_calls(fake_ray):
    actor = _make_actor(_rows_payload(_AddOne))
    first = _data_blocks(list(actor.run_block_stream(pa.table({"x": [1], "keep": ["a"]}))))
    executor_after_first = actor.executor
    second = _data_blocks(list(actor.run_block_stream(pa.table({"x": [5], "keep": ["b"]}))))

    assert actor.executor is executor_after_first
    assert first[0].to_pydict() == {"keep": ["a"], "y": [2]}
    assert second[0].to_pydict() == {"keep": ["b"], "y": [6]}


def test_actor_close_executor_is_terminal(fake_ray):
    payload = _rows_payload(_AddOne)
    actor = _make_actor(payload)

    actor.close_executor()
    actor.close_executor()

    assert actor.executor is None
    with pytest.raises(RuntimeError, match="actor executor is closed"):
        actor._ensure_executor(payload)
    with pytest.raises(RuntimeError, match="actor executor is closed"):
        actor.init_payload(payload)


def test_actor_close_executor_retains_failed_executor_for_retry(fake_ray):
    payload = _rows_payload(_AddOne)
    actor = _make_actor(payload)
    executor = actor.executor
    close_calls = 0

    def transient_close():
        nonlocal close_calls
        close_calls += 1
        if close_calls == 1:
            raise RuntimeError("planned actor executor close failure")

    executor.close = transient_close

    with pytest.raises(RuntimeError, match="planned actor executor close failure"):
        actor.close_executor()

    assert actor.executor is executor
    assert actor._executor_closed is False
    with pytest.raises(RuntimeError, match="actor executor is closed"):
        actor._ensure_executor(payload)

    actor.close_executor()

    assert close_calls == 2
    assert actor.executor is None
    assert actor._executor_closed is True


def test_reconstructed_actor_reconciles_new_node_before_user_code(fake_ray, monkeypatch):
    from vane.runners.ray.query_runtime_protocol import (
        RAY_ACTOR_GENERATION_CAPABILITY_ENV,
        RAY_ACTOR_INDEX_ENV,
        RAY_ACTOR_POOL_NONCE_ENV,
        RAY_ACTOR_QUERY_ID_ENV,
        RAY_ACTOR_RESOURCE_UNIT_ID_ENV,
    )

    payload = _rows_payload(_AddOne)
    payload["actor_index"] = 0
    fake_ray.node_id = "node-after-restart"
    fake_ray.reconstructed = True
    monkeypatch.setenv(RAY_ACTOR_QUERY_ID_ENV, payload["query_id"])
    monkeypatch.setenv(RAY_ACTOR_RESOURCE_UNIT_ID_ENV, payload["resource_unit_id"])
    monkeypatch.setenv(RAY_ACTOR_INDEX_ENV, "0")
    monkeypatch.setenv(RAY_ACTOR_POOL_NONCE_ENV, "pool-nonce")
    monkeypatch.setenv(RAY_ACTOR_GENERATION_CAPABILITY_ENV, "generation-capability")
    actor = _make_actor(payload)

    blocks = _data_blocks(list(actor.run_block_stream(pa.table({"x": [1], "keep": ["a"]}))))

    assert blocks[0].to_pydict() == {"keep": ["a"], "y": [2]}
    assert len(fake_ray.location_reports) == 2
    report, capability = fake_ray.location_reports[-1]
    assert report == {
        "query_id": payload["query_id"],
        "resource_unit_id": payload["resource_unit_id"],
        "actor_index": 0,
        "pool_nonce": "pool-nonce",
        "node_id": "node-after-restart",
    }
    assert capability == "generation-capability"


def test_reconstructed_actor_reconciles_new_node_before_ref_bundle_materialization(fake_ray, monkeypatch):
    import vane.execution.udf_ray_actor_runtime as actor_runtime
    from vane.runners.ray.query_runtime_protocol import (
        RAY_ACTOR_GENERATION_CAPABILITY_ENV,
        RAY_ACTOR_INDEX_ENV,
        RAY_ACTOR_POOL_NONCE_ENV,
        RAY_ACTOR_QUERY_ID_ENV,
        RAY_ACTOR_RESOURCE_UNIT_ID_ENV,
    )

    payload = _rows_payload(_AddOne)
    payload["actor_index"] = 0
    fake_ray.node_id = "node-after-restart"
    fake_ray.reconstructed = True
    monkeypatch.setenv(RAY_ACTOR_QUERY_ID_ENV, payload["query_id"])
    monkeypatch.setenv(RAY_ACTOR_RESOURCE_UNIT_ID_ENV, payload["resource_unit_id"])
    monkeypatch.setenv(RAY_ACTOR_INDEX_ENV, "0")
    monkeypatch.setenv(RAY_ACTOR_POOL_NONCE_ENV, "pool-nonce")
    monkeypatch.setenv(RAY_ACTOR_GENERATION_CAPABILITY_ENV, "generation-capability")
    actor = _make_actor(payload)
    original_materialize = actor_runtime._apply_ref_bundle_slices

    def materialize_after_reconciliation(*args, **kwargs):
        assert len(fake_ray.location_reports) == 2
        return original_materialize(*args, **kwargs)

    monkeypatch.setattr(actor_runtime, "_apply_ref_bundle_slices", materialize_after_reconciliation)
    block = pa.table({"x": [1], "keep": ["a"]})

    blocks = _data_blocks(
        list(
            actor.run_ref_bundle_stream(
                block,
                slices=[(0, 1)],
                metadata=[{"num_rows": 1}],
                names=["x", "keep"],
            )
        )
    )

    assert blocks[0].to_pydict() == {"keep": ["a"], "y": [2]}
    assert len(fake_ray.location_reports) == 2
