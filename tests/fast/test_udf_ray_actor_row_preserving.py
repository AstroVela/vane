# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Ray actor block streams must split/fuse row-preserving inputs."""

from __future__ import annotations

import sys
import types
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
        "stage_id": "stage-row-preserving",
        "task_lease_id": "lease-row-preserving",
        "attempt_id": "attempt-row-preserving",
        "node_id": "node-row-preserving",
        "udf_output_target_max_bytes": 1 << 20,
        "output_window_bytes": 2 << 20,
    }


class _FakeRuntimeContext:
    def get_node_id(self) -> str:
        return "node-row-preserving"


class _FakeRayModule(types.ModuleType):
    def __init__(self) -> None:
        super().__init__("ray")

    def remote(self, *args: Any, **kwargs: Any):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]

        def deco(cls: Any) -> Any:
            return cls

        return deco

    def get_runtime_context(self) -> _FakeRuntimeContext:
        return _FakeRuntimeContext()


@pytest.fixture()
def fake_ray(monkeypatch: pytest.MonkeyPatch) -> _FakeRayModule:
    # Load the actor runtime and its runner dependencies against real Ray.
    # The fake only needs to replace Ray while the actor class itself runs.
    import vane.execution.udf_ray_actor_runtime  # noqa: F401

    module = _FakeRayModule()
    monkeypatch.setitem(sys.modules, "ray", module)
    return module


class _AddOne:
    def __call__(self, table: pa.Table) -> pa.Table:
        return pa.table({"y": [value + 1 for value in table.column("x").to_pylist()]})


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
