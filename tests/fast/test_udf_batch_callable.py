# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""BatchUDF must preserve the actor/compute/output boundary and ownership."""

from __future__ import annotations

import gc
import threading
import weakref
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pyarrow as pa
import pytest

from vane.execution._udf_runtime import UDFExecutor
from vane.execution.udf_output_schema import batch_udf_output_schema, columns_to_output_table
from vane.pickle import dumps
from vane.udf import BatchUDF


def _payload(udf, **overrides):
    return {
        "function_pickle": dumps(udf),
        "udf_name": udf.__name__,
        "call_mode": "map_batches",
        "execution_backend": "ray_actor",
        "batch_size": 2,
        "output_schema": [{"name": "y", "type": "BIGINT"}],
        **overrides,
    }


class _RecordingUDF(BatchUDF):
    def __init__(self):
        self.owner = threading.get_ident()
        self.events = []

    def warm_up(self):
        assert threading.get_ident() == self.owner

    def prepare_batch(self, table):
        assert threading.get_ident() == self.owner
        self.events.append(("prepare", table.num_rows, threading.get_ident()))
        return table.column(0).to_pylist()

    def __call__(self, values):
        assert threading.get_ident() != self.owner
        self.events.append(("compute", len(values), threading.get_ident()))
        return {"y": [value + 1 for value in values]}

    def _vane_close(self):
        assert threading.get_ident() == self.owner
        self.events.append(("close", 0, threading.get_ident()))


@pytest.mark.parametrize("stream_output", [False, True])
@pytest.mark.parametrize("preserve", [False, True])
def test_batch_udf_boundary_order_and_tail(monkeypatch, stream_output, preserve):
    import vane.execution.udf_batch_callable as boundary

    owner = threading.get_ident()
    encode = boundary.columns_to_output_table
    encoder_threads = []

    def record_encode(*args, **kwargs):
        encoder_threads.append(threading.get_ident())
        return encode(*args, **kwargs)

    monkeypatch.setattr(boundary, "columns_to_output_table", record_encode)
    runtime = UDFExecutor(
        _payload(
            _RecordingUDF, stream_output=stream_output, output_batch_size=3, preserve_compute_batch_boundaries=preserve
        )
    )
    udf = runtime._map_fn
    try:
        runtime.warm_up()
        outputs = list(runtime.iter_submit(pa.table({"x": [1, 2, 3, 4, 5]})))
        assert pa.concat_tables(outputs).to_pydict() == {"y": [2, 3, 4, 5, 6]}
        if stream_output:
            assert [table.num_rows for table in outputs] == ([2, 2, 1] if preserve else [3, 2])
        assert [(kind, rows) for kind, rows, _ in udf.events] == [
            ("prepare", 2),
            ("compute", 2),
            ("prepare", 2),
            ("compute", 2),
            ("prepare", 1),
            ("compute", 1),
        ]
        assert encoder_threads == [owner] * 3
        compute_threads = {tid for kind, _, tid in udf.events if kind == "compute"}
        assert len(compute_threads) == 1
        assert owner not in compute_threads
    finally:
        runtime.close()
    runtime.close()
    assert udf.events[-1] == ("close", 0, owner)
    assert not any(thread.ident in compute_threads and thread.is_alive() for thread in threading.enumerate())
    with pytest.raises(RuntimeError, match="closing or closed"):
        list(runtime.iter_submit(pa.table({"x": [1]})))


def _video_schema():
    return batch_udf_output_schema(
        {
            "output_schema": [
                {"name": "frame_index", "type": "BIGINT"},
                {"name": "frame", "kind": "tensor", "dtype": "UTINYINT", "shape": [2, 3, 3]},
                {"name": "features", "type": "STRUCT(label BIGINT, confidence DOUBLE, bbox DOUBLE[])[]"},
            ]
        }
    )


@pytest.mark.parametrize("rows", [0, 1, 3])
def test_batch_udf_video_schema_empty_detections_and_zero_copy_frames(rows):
    schema = _video_schema()
    frames = np.arange(rows * 18, dtype=np.uint8).reshape(rows, 2, 3, 3)
    columns = {"features": [[] for _ in range(rows)], "frame": frames, "frame_index": list(range(rows))}
    table = columns_to_output_table(columns, schema, udf_name="video")
    assert table.schema == schema
    assert table.num_rows == rows
    assert table.column("features").to_pylist() == [[] for _ in range(rows)]
    if rows:
        actual = table.column("frame").chunk(0).to_numpy_ndarray()
        assert np.shares_memory(actual, frames)
        np.testing.assert_array_equal(actual, frames)


def test_batch_udf_video_nested_values_preserve_types():
    features = [[{"label": 2, "confidence": 0.125, "bbox": [1.0, 2.0, 3.0, 4.0]}], []]
    table = columns_to_output_table(
        {"frame_index": [0, 1], "frame": np.zeros((2, 2, 3, 3), dtype=np.uint8), "features": features},
        _video_schema(),
        udf_name="video",
    )
    assert table.column("features").to_pylist() == features


@pytest.mark.parametrize("stream_output", [False, True])
def test_batch_udf_nullable_nested_output_matches_ordinary_contract(stream_output):
    features = [None, [], [None], [{"label": None, "confidence": None, "bbox": [None, 2.0]}]]
    entries = [
        {"name": "frame_index", "type": "BIGINT"},
        {"name": "frame", "kind": "tensor", "dtype": "UTINYINT", "shape": [2, 3, 3]},
        {"name": "features", "type": "STRUCT(label BIGINT, confidence DOUBLE, bbox DOUBLE[])[]"},
    ]

    class Materialized(BatchUDF):
        def __call__(self, table):
            pixels = np.arange(4 * 18, dtype=np.uint8).reshape(4, 2, 3, 3)
            pixels.flags.writeable = False
            return {"frame_index": [0, 1, 2, 3], "frame": pixels, "features": features}

    class Ordinary:
        def __call__(self, table):
            return columns_to_output_table(Materialized()(table), _video_schema(), udf_name="reference")

    outputs = []
    for udf in (Ordinary, Materialized):
        runtime = UDFExecutor(_payload(udf, output_schema=entries, batch_size=4, stream_output=stream_output))
        try:
            outputs.append(pa.concat_tables(list(runtime.iter_submit(pa.table({"x": [0, 1, 2, 3]})))))
        finally:
            runtime.close()
    assert outputs[0].equals(outputs[1], check_metadata=True)
    assert outputs[1].column("features").to_pylist() == features
    np.testing.assert_array_equal(
        outputs[1].column("frame").chunk(0).to_numpy_ndarray(), np.arange(4 * 18, dtype=np.uint8).reshape(4, 2, 3, 3)
    )


def test_batch_udf_retains_normalization_for_different_logical_contract():
    import vane

    class Output(BatchUDF):
        def __call__(self, table):
            return {"frame": np.zeros((1, 2, 3, 3), dtype=np.uint8), "record": [{"x": 7}]}

    runtime = UDFExecutor(
        _payload(
            Output,
            output_schema=[
                {"name": "frame", "kind": "tensor", "dtype": "UTINYINT", "shape": [2, 3, 3]},
                {"name": "record", "type": "STRUCT(x BIGINT)"},
            ],
            output_contract_types=[str(vane.tensor_type(vane.sqltypes.UTINYINT, [2, 3, 3])), "STRUCT(X BIGINT)"],
        )
    )
    try:
        output = list(runtime.iter_submit(pa.table({"x": [1]})))[0]
        assert output.column("record").type == pa.struct([("X", pa.int64())])
        assert output.column("record").to_pylist() == [{"X": 7}]
    finally:
        runtime.close()


def test_batch_udf_retains_governed_output_validation():
    import vane

    class Output(BatchUDF):
        def __call__(self, table):
            return {"y": [1]}

    runtime = UDFExecutor(_payload(Output, output_contract_types=["FILE"]))
    try:
        with pytest.raises(vane.InvalidInputException, match="FILE"):
            list(runtime.iter_submit(pa.table({"x": [1]})))
    finally:
        runtime.close()


@pytest.mark.parametrize(
    "bad_frame",
    [
        np.zeros((1, 2, 3, 3), dtype=np.float32),
        np.zeros((1, 3, 2, 3), dtype=np.uint8),
        np.zeros((1, 2, 3, 6), dtype=np.uint8)[..., ::2],
        [],
    ],
)
def test_batch_udf_rejects_tensor_coercion(bad_frame):
    with pytest.raises((TypeError, ValueError), match="column 'frame'"):
        columns_to_output_table(
            {"frame_index": [0], "frame": bad_frame, "features": [[]]}, _video_schema(), udf_name="video"
        )


@pytest.mark.parametrize(
    "value,match",
    [
        ({"other": [1]}, "declared columns"),
        ({"y": pa.array([1])}, "materialized"),
        ({"y": np.array(1)}, "one-dimensional"),
        ({"y": np.zeros((2, 2))}, "one-dimensional"),
        ({"y": [2**100]}, "could not encode"),
        (None, "materialized dict"),
        (pa.table({"y": [1]}), "materialized dict"),
    ],
)
def test_batch_udf_rejects_invalid_columns(value, match):
    with pytest.raises((TypeError, ValueError), match=match):
        columns_to_output_table(value, pa.schema([("y", pa.int64())]), udf_name="invalid")


def test_batch_udf_rejects_mismatched_column_lengths_and_does_not_print_values():
    with pytest.raises(ValueError, match="has 2 rows, expected 1"):
        columns_to_output_table(
            {"a": [1], "b": [1, 2]}, pa.schema([("a", pa.int64()), ("b", pa.int64())]), udf_name="test"
        )
    with pytest.raises(ValueError) as error:
        columns_to_output_table({"y": ["private-pixel-content"]}, pa.schema([("y", pa.int64())]), udf_name="test")
    assert "private-pixel-content" not in str(error.value)


@pytest.mark.parametrize(
    "overrides",
    [
        {"execution_backend": "subprocess_actor"},
        {"execution_backend": "ray_task"},
        {"call_mode": "map_batches_rows"},
        {"call_mode": "flat_map"},
        {"call_mode": "map"},
        {"row_preserving": True},
        {"output_schema": [{"name": "y", "type": "FILE"}]},
        {"output_schema": [{"name": "y", "kind": "tensor", "dtype": "UTINYINT", "shape": [None, 3]}]},
        {"output_schema": [{"name": "y", "type": "DATE"}]},
        {"output_schema": [{"name": "y", "type": "BIGINT"}, {"name": "Y", "type": "BIGINT"}]},
    ],
)
def test_batch_udf_unsupported_contract_fails_before_construction(overrides):
    class NeverConstruct(BatchUDF):
        def __init__(self):
            raise AssertionError("constructor must not run")

        def __call__(self, table):
            return {"y": []}

    with pytest.raises((TypeError, ValueError)):
        UDFExecutor(_payload(NeverConstruct, **overrides))


@pytest.mark.parametrize("method", ["prepare_batch", "__call__"])
def test_batch_udf_generator_methods_fail_before_construction(method):
    def gen(self, value):
        yield value

    cls = type("GeneratorUDF", (_RecordingUDF,), {method: gen})
    with pytest.raises(TypeError, match="generator function"):
        UDFExecutor(_payload(cls))


def test_batch_udf_does_not_consume_dynamic_generator_result():
    class GeneratorResult(BatchUDF):
        def __call__(self, table):
            def outputs():
                raise AssertionError("must not iterate on actor thread")
                yield

            return outputs()

    runtime = UDFExecutor(_payload(GeneratorResult))
    try:
        with pytest.raises(TypeError, match="materialized dict"):
            runtime.submit(pa.table({"x": [1]}))
    finally:
        runtime.close()


def test_batch_udf_awaitable_result_is_closed():
    class AwaitableResult(BatchUDF):
        def __call__(self, table):
            async def output():
                return {"y": [1]}

            return output()

    runtime = UDFExecutor(_payload(AwaitableResult))
    try:
        with pytest.raises(TypeError, match="return values synchronously"):
            runtime.submit(pa.table({"x": [1]}))
    finally:
        runtime.close()


def test_batch_udf_output_owns_numpy_after_worker_shutdown():
    class Arrays(BatchUDF):
        def __init__(self):
            self.outputs = []

        def __call__(self, table):
            array = np.arange(table.num_rows, dtype=np.int64)
            self.outputs.append(weakref.ref(array))
            return {"y": array}

    runtime = UDFExecutor(_payload(Arrays, stream_output=True, preserve_compute_batch_boundaries=True))
    udf = runtime._map_fn
    outputs = list(runtime.iter_submit(pa.table({"x": [1, 2, 3, 4]})))
    runtime.close()
    assert all(ref() is not None for ref in udf.outputs)
    assert [table.to_pydict() for table in outputs] == [{"y": [0, 1]}, {"y": [0, 1]}]
    del outputs
    gc.collect()
    assert all(ref() is None for ref in udf.outputs)


def test_batch_udf_backpressure_does_not_schedule_ahead():
    runtime = UDFExecutor(_payload(_RecordingUDF, stream_output=True, preserve_compute_batch_boundaries=True))
    outputs = runtime.iter_submit(pa.table({"x": list(range(10))}))
    try:
        assert next(outputs).column("y").to_pylist() == [1, 2]
        assert [kind for kind, _, _ in runtime._map_fn.events] == ["prepare", "compute"]
    finally:
        outputs.close()
        runtime.close()


@pytest.mark.parametrize("failure", ["prepare", "compute", "encode"])
def test_batch_udf_errors_allow_worker_cleanup(failure):
    class Broken(BatchUDF):
        def prepare_batch(self, table):
            if failure == "prepare":
                raise ValueError("failed-prepare")
            return table

        def __call__(self, table):
            if failure == "compute":
                raise ValueError("failed-compute")
            return {"y": ["cannot encode integer"]}

    before = set(threading.enumerate())
    runtime = UDFExecutor(_payload(Broken))
    try:
        with pytest.raises(ValueError):
            runtime.submit(pa.table({"x": [1]}))
    finally:
        runtime.close()
    assert not [t for t in set(threading.enumerate()) - before if t.name.startswith("vane-batch-udf")]


def test_batch_udf_close_failure_still_joins_worker_and_can_retry():
    class BrokenClose(_RecordingUDF):
        def _vane_close(self):
            super()._vane_close()
            if sum(kind == "close" for kind, _, _ in self.events) == 1:
                raise ValueError("close once")

    runtime = UDFExecutor(_payload(BrokenClose))
    runtime.submit(pa.table({"x": [1]}))
    with pytest.raises(RuntimeError, match="close once"):
        runtime.close()
    assert runtime._batch_callable._pool is None
    with pytest.raises(RuntimeError, match="closing or closed"):
        runtime.submit(pa.table({"x": [1]}))
    runtime.close()
    assert runtime._map_fn is None


def test_batch_udf_rejects_close_from_another_thread_without_poisoning_owner():
    runtime = UDFExecutor(_payload(_RecordingUDF))
    with ThreadPoolExecutor(max_workers=1) as other:
        with pytest.raises(RuntimeError, match="owning actor thread"):
            other.submit(runtime.close).result()
    runtime.submit(pa.table({"x": [1]}))
    runtime.close()


def test_batch_udf_instances_do_not_share_workers(monkeypatch):
    monkeypatch.setenv("VANE_CPU_UDF_CALLABLE_CACHE", "1")
    payload = _payload(_RecordingUDF)
    first = UDFExecutor(payload, cache_callable=True)
    second = UDFExecutor(payload, cache_callable=True)
    try:
        first.submit(pa.table({"x": [1]}))
        second.submit(pa.table({"x": [2]}))
        assert first._map_fn.events[1][2] != second._map_fn.events[1][2]
        first.close()
        second.submit(pa.table({"x": [3]}))
        assert len(second._map_fn.events) == 4
    finally:
        first.close()
        second.close()


@pytest.mark.parametrize("method", ["prepare_batch", "warm_up", "_vane_close"])
def test_batch_udf_async_hooks_are_rejected(method):
    async def async_hook(self, *args):
        return None

    cls = type("AsyncHook", (_RecordingUDF,), {method: async_hook})
    with pytest.raises(TypeError, match="must be synchronous"):
        UDFExecutor(_payload(cls))


@pytest.mark.parametrize("method", ["warm_up", "_vane_close"])
def test_batch_udf_dynamic_awaitable_hooks_are_closed(method):
    async def async_hook():
        return None

    def hook(self):
        return async_hook()

    cls = type("DynamicAsyncHook", (_RecordingUDF,), {method: hook})
    runtime = UDFExecutor(_payload(cls))
    try:
        action = runtime.warm_up if method == "warm_up" else runtime.close
        with pytest.raises((TypeError, RuntimeError), match="return values synchronously"):
            action()
    finally:
        runtime._map_fn._vane_close = lambda: None
        runtime.close()


def test_batch_udf_empty_input_does_not_start_worker():
    runtime = UDFExecutor(_payload(_RecordingUDF, stream_output=True))
    try:
        assert list(runtime.iter_submit(pa.table({"x": pa.array([], type=pa.int64())}))) == []
        assert runtime._map_fn.events == []
        assert runtime._batch_callable._pool is None
    finally:
        runtime.close()


@pytest.mark.parametrize("stream_output", [False, True])
def test_batch_udf_can_change_row_count(stream_output):
    class Expand(BatchUDF):
        def __call__(self, table):
            return {"y": [table.num_rows] * 3}

    runtime = UDFExecutor(_payload(Expand, stream_output=stream_output))
    try:
        output = pa.concat_tables(list(runtime.iter_submit(pa.table({"x": [1, 2, 3]}))))
        assert output.column("y").to_pylist() == [2, 2, 2, 1, 1, 1]
    finally:
        runtime.close()


def test_plain_callable_prepare_method_does_not_opt_in():
    class Ordinary:
        def __init__(self):
            self.owner = threading.get_ident()

        def prepare_batch(self, table):
            raise AssertionError("ordinary callable must not use BatchUDF protocol")

        def __call__(self, table):
            assert threading.get_ident() == self.owner
            return {"y": table.column(0)}

    runtime = UDFExecutor(_payload(Ordinary))
    try:
        assert runtime._batch_callable is None
        assert list(runtime.iter_submit(pa.table({"x": [1]})))[0].to_pydict() == {"y": [1]}
    finally:
        runtime.close()


@pytest.mark.real_ray
def test_batch_udf_real_actor_public_map_batches(ray_local, monkeypatch):
    import vane

    monkeypatch.setenv("VANE_RUNNER", "ray")
    with vane.connect(config={"threads": 2}) as con:
        output = (
            con.sql("select range as x from range(5)")
            .map_batches(
                _RecordingUDF,
                schema={"y": vane.sqltypes.BIGINT},
                batch_size=2,
                execution_backend="ray_actor",
                actor_number=1,
            )
            .fetchall()
        )
    assert sorted(output) == [(1,), (2,), (3,), (4,), (5,)]
