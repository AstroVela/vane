# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Ordinary callable-class actors preserve computation/output boundaries and ownership."""

from __future__ import annotations

import gc
import os
import subprocess
import sys
import textwrap
import threading
import weakref
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pyarrow as pa
import pytest

from vane.execution._udf_runtime import UDFExecutor
from vane.execution.udf_output_schema import columns_to_output_table, materialized_output_schema
from vane.pickle import dumps


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


class _RecordingUDF:
    def __init__(self):
        self.owner = threading.get_ident()
        self.events = []

    def warm_up(self):
        assert threading.get_ident() == self.owner

    def __call__(self, table):
        values = table.column(0).to_pylist()
        assert threading.get_ident() != self.owner
        self.events.append(("compute", len(values), threading.get_ident()))
        return {"y": [value + 1 for value in values]}

    def _vane_close(self):
        assert threading.get_ident() == self.owner
        self.events.append(("close", 0, threading.get_ident()))


@pytest.mark.parametrize("backend", ["ray_actor", "subprocess_actor"])
@pytest.mark.parametrize("stream_output", [False, True])
@pytest.mark.parametrize("preserve", [False, True])
def test_actor_udf_boundary_order_and_tail(monkeypatch, backend, stream_output, preserve):
    owner = threading.get_ident()
    encode = pa.table
    encoder_threads = []

    def record_encode(*args, **kwargs):
        if isinstance(args[0], dict) and set(args[0]) == {"y"}:
            encoder_threads.append(threading.get_ident())
        return encode(*args, **kwargs)

    monkeypatch.setattr(pa, "table", record_encode)
    runtime = UDFExecutor(
        _payload(
            _RecordingUDF,
            execution_backend=backend,
            stream_output=stream_output,
            output_batch_size=3,
            preserve_compute_batch_boundaries=preserve,
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
            ("compute", 2),
            ("compute", 2),
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
    return materialized_output_schema(
        {
            "output_schema": [
                {"name": "frame_index", "type": "BIGINT"},
                {"name": "frame", "kind": "tensor", "dtype": "UTINYINT", "shape": [2, 3, 3]},
                {"name": "features", "type": "STRUCT(label BIGINT, confidence DOUBLE, bbox DOUBLE[])[]"},
            ]
        }
    )


@pytest.mark.parametrize("rows", [0, 1, 3])
def test_actor_udf_video_schema_empty_detections_and_zero_copy_frames(rows):
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


def test_actor_udf_video_nested_values_preserve_types():
    features = [[{"label": 2, "confidence": 0.125, "bbox": [1.0, 2.0, 3.0, 4.0]}], []]
    table = columns_to_output_table(
        {"frame_index": [0, 1], "frame": np.zeros((2, 2, 3, 3), dtype=np.uint8), "features": features},
        _video_schema(),
        udf_name="video",
    )
    assert table.column("features").to_pylist() == features


@pytest.mark.parametrize("stream_output", [False, True])
def test_actor_udf_nullable_nested_output_matches_ordinary_contract(stream_output):
    features = [None, [], [None], [{"label": None, "confidence": None, "bbox": [None, 2.0]}]]
    entries = [
        {"name": "frame_index", "type": "BIGINT"},
        {"name": "frame", "kind": "tensor", "dtype": "UTINYINT", "shape": [2, 3, 3]},
        {"name": "features", "type": "STRUCT(label BIGINT, confidence DOUBLE, bbox DOUBLE[])[]"},
    ]

    class Materialized:
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


def test_actor_udf_retains_normalization_for_different_logical_contract():
    import vane

    class Output:
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


def test_actor_udf_retains_governed_output_validation():
    import vane

    class Output:
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
def test_actor_udf_rejects_tensor_coercion(bad_frame):
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
def test_actor_udf_rejects_invalid_columns(value, match):
    with pytest.raises((TypeError, ValueError), match=match):
        columns_to_output_table(value, pa.schema([("y", pa.int64())]), udf_name="invalid")


def test_actor_udf_rejects_mismatched_column_lengths_and_does_not_print_values():
    with pytest.raises(ValueError, match="has 2 rows, expected 1"):
        columns_to_output_table(
            {"a": [1], "b": [1, 2]}, pa.schema([("a", pa.int64()), ("b", pa.int64())]), udf_name="test"
        )
    with pytest.raises(ValueError) as error:
        columns_to_output_table({"y": ["private-pixel-content"]}, pa.schema([("y", pa.int64())]), udf_name="test")
    assert "private-pixel-content" not in str(error.value)


@pytest.mark.parametrize("dynamic", [False, True])
def test_actor_udf_generator_remains_lazy_and_runs_on_owner(dynamic):
    class Generator:
        def __init__(self):
            self.owner = threading.get_ident()
            self.calls = 0

        def outputs(self, table):
            assert threading.get_ident() == self.owner
            self.calls += 1
            yield {"y": table.column(0)}
            self.calls += 1
            raise ValueError("later output")

        if dynamic:

            def __call__(self, table):
                assert threading.get_ident() != self.owner
                return self.outputs(table)
        else:

            def __call__(self, table):
                yield from self.outputs(table)

    runtime = UDFExecutor(_payload(Generator, stream_output=True, output_batch_size=1))
    outputs = runtime.iter_submit(pa.table({"x": [1]}))
    try:
        assert next(outputs).to_pydict() == {"y": [1]}
        assert runtime._map_fn.calls == 1
        with pytest.raises(ValueError, match="later output"):
            next(outputs)
        assert runtime._map_fn.calls == 2
    finally:
        outputs.close()
        runtime.close()


def test_actor_udf_awaitable_result_is_closed():
    class AwaitableResult:
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


def test_actor_udf_output_owns_numpy_after_worker_shutdown():
    class Arrays:
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


def test_actor_udf_backpressure_does_not_schedule_ahead():
    runtime = UDFExecutor(_payload(_RecordingUDF, stream_output=True, preserve_compute_batch_boundaries=True))
    outputs = runtime.iter_submit(pa.table({"x": list(range(10))}))
    try:
        assert next(outputs).column("y").to_pylist() == [1, 2]
        assert [kind for kind, _, _ in runtime._map_fn.events] == ["compute"]
    finally:
        outputs.close()
        runtime.close()


@pytest.mark.parametrize("failure", ["prepare", "compute", "encode"])
def test_actor_udf_errors_allow_worker_cleanup(monkeypatch, failure):
    from vane.execution.udf_file_contract import FileUDFContract

    class Broken:
        def __call__(self, table):
            if failure == "compute":
                raise ValueError("failed-compute")
            return {"y": [object()]}

    if failure == "prepare":

        def fail_prepare(self, table):
            raise ValueError("failed-prepare")

        monkeypatch.setattr(FileUDFContract, "prepare_input_table", fail_prepare)

    before = set(threading.enumerate())
    runtime = UDFExecutor(_payload(Broken))
    try:
        with pytest.raises(ValueError):
            runtime.submit(pa.table({"x": [1]}))
    finally:
        runtime.close()
    assert not [t for t in set(threading.enumerate()) - before if t.name.startswith("vane-actor-udf")]


def test_actor_udf_close_failure_still_joins_worker_and_can_retry():
    class BrokenClose(_RecordingUDF):
        def _vane_close(self):
            super()._vane_close()
            if sum(kind == "close" for kind, _, _ in self.events) == 1:
                raise ValueError("close once")

    runtime = UDFExecutor(_payload(BrokenClose))
    runtime.submit(pa.table({"x": [1]}))
    with pytest.raises(RuntimeError, match="close once"):
        runtime.close()
    assert runtime._actor_callable._pool is None
    with pytest.raises(RuntimeError, match="closing or closed"):
        runtime.submit(pa.table({"x": [1]}))
    runtime.close()
    assert runtime._map_fn is None


def test_actor_udf_rejects_close_from_another_thread_without_poisoning_owner():
    runtime = UDFExecutor(_payload(_RecordingUDF))
    with ThreadPoolExecutor(max_workers=1) as other:
        with pytest.raises(RuntimeError, match="owning actor thread"):
            other.submit(runtime.close).result()
    runtime.submit(pa.table({"x": [1]}))
    runtime.close()


def test_actor_udf_instances_do_not_share_workers(monkeypatch):
    monkeypatch.setenv("VANE_CPU_UDF_CALLABLE_CACHE", "1")
    payload = _payload(_RecordingUDF)
    first = UDFExecutor(payload, cache_callable=True)
    second = UDFExecutor(payload, cache_callable=True)
    try:
        first.submit(pa.table({"x": [1]}))
        second.submit(pa.table({"x": [2]}))
        assert first._map_fn.events[0][2] != second._map_fn.events[0][2]
        first.close()
        second.submit(pa.table({"x": [3]}))
        assert len(second._map_fn.events) == 2
    finally:
        first.close()
        second.close()


@pytest.mark.parametrize("method", ["warm_up", "_vane_close"])
def test_actor_udf_dynamic_awaitable_hooks_are_closed(method):
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


def test_actor_udf_empty_input_does_not_start_worker():
    runtime = UDFExecutor(_payload(_RecordingUDF, stream_output=True))
    try:
        assert list(runtime.iter_submit(pa.table({"x": pa.array([], type=pa.int64())}))) == []
        assert runtime._map_fn.events == []
        assert runtime._actor_callable._pool is None
    finally:
        runtime.close()


@pytest.mark.parametrize("stream_output", [False, True])
def test_actor_udf_can_change_row_count(stream_output):
    class Expand:
        def __call__(self, table):
            return {"y": [table.num_rows] * 3}

    runtime = UDFExecutor(_payload(Expand, stream_output=stream_output))
    try:
        output = pa.concat_tables(list(runtime.iter_submit(pa.table({"x": [1, 2, 3]}))))
        assert output.column("y").to_pylist() == [2, 2, 2, 1, 1, 1]
    finally:
        runtime.close()


def test_plain_callable_prepare_method_is_not_a_hook():
    class Ordinary:
        def __init__(self):
            self.owner = threading.get_ident()

        def prepare_batch(self, table):
            raise AssertionError("prepare_batch is not a framework hook")

        def __call__(self, table):
            assert threading.get_ident() != self.owner
            return {"y": table.column(0)}

    runtime = UDFExecutor(_payload(Ordinary))
    try:
        assert runtime._actor_callable is not None
        assert list(runtime.iter_submit(pa.table({"x": [1]})))[0].to_pydict() == {"y": [1]}
    finally:
        runtime.close()


@pytest.mark.parametrize("backend", ["ray_actor", "subprocess_actor", "ray_task", "subprocess_task"])
def test_numpy_tensor_encoding_runs_on_owner_and_validates_output(monkeypatch, backend):
    import vane.execution.udf_output_schema as schema_module
    from vane.execution.udf_file_contract import FileUDFContract

    owner = threading.get_ident()
    events = []
    encode = schema_module.columns_to_output_table
    validate = FileUDFContract.validate_output_table

    def record_encode(*args, **kwargs):
        events.append(("encode", threading.get_ident()))
        return encode(*args, **kwargs)

    def record_validate(self, table):
        events.append(("validate", threading.get_ident()))
        return validate(self, table)

    monkeypatch.setattr(schema_module, "columns_to_output_table", record_encode)
    monkeypatch.setattr(FileUDFContract, "validate_output_table", record_validate)

    def output(table):
        return {"y": np.zeros((table.num_rows, 2), dtype=np.int64)}

    class Output:
        def __call__(self, table):
            return output(table)

    runtime = UDFExecutor(
        _payload(
            Output if backend.endswith("actor") else output,
            execution_backend=backend,
            output_schema=[{"name": "y", "kind": "tensor", "dtype": "BIGINT", "shape": [2]}],
        )
    )
    try:
        result = list(runtime.iter_submit(pa.table({"x": [1]})))[0]
        assert isinstance(result.column(0).type, pa.FixedShapeTensorType)
        assert events == [("encode", owner), ("validate", owner)]
    finally:
        runtime.close()


def test_existing_dict_output_preserves_inference_and_column_order():
    class Output:
        def __call__(self, table):
            return {"other": pa.array([1.75]), "value": [2.5]}

    runtime = UDFExecutor(
        _payload(Output, output_schema=[{"name": "a", "type": "BIGINT"}, {"name": "b", "type": "BIGINT"}])
    )
    try:
        output = list(runtime.iter_submit(pa.table({"x": [1]})))[0]
        assert output.to_pydict() == {"other": [1.75], "value": [2.5]}
        assert output.schema.types == [pa.float64(), pa.float64()]
    finally:
        runtime.close()


@pytest.mark.parametrize("rows", [0, 1])
def test_numpy_tensor_output_without_pandas(rows):
    script = textwrap.dedent("""
        import importlib.abc
        import sys

        class NoPandas(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "pandas" or fullname.startswith("pandas."):
                    raise ModuleNotFoundError("pandas is not installed", name=fullname)

        sys.meta_path.insert(0, NoPandas())
        import numpy as np
        import pyarrow as pa
        from vane.execution._udf_runtime import UDFExecutor
        from vane.pickle import dumps

        rows = int(sys.argv[1])

        class Output:
            def __call__(self, table):
                return {"pixels": np.arange(rows * 6, dtype=np.uint8).reshape(rows, 2, 3)}

        runtime = UDFExecutor({
            "function_pickle": dumps(Output),
            "call_mode": "map_batches",
            "execution_backend": "subprocess_actor",
            "output_schema": [{"name": "pixels", "kind": "tensor", "dtype": "UTINYINT", "shape": [2, 3]}],
        })
        try:
            result = list(runtime.iter_submit(pa.table({"x": [1]})))[0]
            assert result.num_rows == rows
            assert result.column(0).type == pa.fixed_shape_tensor(pa.uint8(), [2, 3])
            if rows:
                np.testing.assert_array_equal(
                    result.column(0).chunk(0).to_numpy_ndarray(), np.arange(6, dtype=np.uint8).reshape(1, 2, 3)
                )
            assert "pandas" not in sys.modules
        finally:
            runtime.close()
    """)
    completed = subprocess.run(
        [sys.executable, "-I", "-c", script, str(rows)], capture_output=True, text=True, timeout=30
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_actor_udf_public_local_map_batches():
    script = textwrap.dedent("""
        import threading
        import vane

        class AddOne:
            def __init__(self):
                self.owner = threading.get_ident()

            def __call__(self, table):
                assert threading.get_ident() != self.owner
                return {"y": [value + 1 for value in table.column("x").to_pylist()]}

        with vane.connect(config={"threads": 2}) as con:
            output = con.sql("select range as x from range(5)").map_batches(
                AddOne, schema={"y": vane.sqltypes.BIGINT}, batch_size=2,
                execution_backend="subprocess_actor", actor_number=1,
            ).fetchall()
        assert sorted(output) == [(1,), (2,), (3,), (4,), (5,)], output
    """)
    completed = subprocess.run(
        [sys.executable, "-I", "-c", script],
        capture_output=True,
        text=True,
        timeout=45,
        env={**os.environ, "VANE_RUNNER": "local-fast"},
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.mark.real_ray
def test_actor_udf_real_actor_public_map_batches(ray_local, monkeypatch):
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
