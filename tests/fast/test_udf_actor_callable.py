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
from collections import UserList, deque
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
    if not rows:
        assert table.schema == schema
    assert table.column("frame").type == schema.field("frame").type
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


@pytest.mark.parametrize("positional", [False, True])
def test_materialized_output_matches_nested_struct_field_names_case_insensitively(positional):
    record_type = pa.struct(
        [
            ("x", pa.int64()),
            ("child", pa.struct([("y", pa.int64())])),
            ("items", pa.list_(pa.struct([("z", pa.int64())]))),
            ("fixed", pa.list_(pa.struct([("w", pa.int64())]), 1)),
        ]
    )
    row = {"X": 7, "CHILD": {"Y": 11}, "Items": [{"Z": 13}, None], "FIXED": [{"W": 17}]}
    null_row = {name: None for name in row}
    values = [tuple(row.values()) if positional else row, None, tuple(null_row.values()) if positional else null_row]
    schema = pa.schema([("tensor", pa.fixed_shape_tensor(pa.int64(), [3])), ("record", record_type)])
    output = columns_to_output_table(
        {"tensor": np.arange(9).reshape(3, 3), "record": values}, schema, udf_name="casefold"
    )

    assert output.column("tensor").type == schema.field("tensor").type
    assert output.column("record").to_pylist() == [
        {"x": 7, "child": {"y": 11}, "items": [{"z": 13}, None], "fixed": [{"w": 17}]},
        None,
        {"x": None, "child": None, "items": None, "fixed": None},
    ]
    assert row["CHILD"] == {"Y": 11}


@pytest.mark.parametrize("nested", [False, True])
def test_materialized_output_rejects_ambiguous_struct_field_names(nested):
    import vane

    dtype = pa.struct([("x", pa.int64())])
    value = {"x": 7, "X": 8}
    if nested:
        dtype = pa.list_(dtype)
        value = [value]
    with pytest.raises(vane.InvalidInputException, match="ambiguous"):
        columns_to_output_table({"record": [value]}, pa.schema([("record", dtype)]), udf_name="ambiguous")


@pytest.mark.parametrize("value", [{"X": 7, "extra": "private-extra-value"}, {}, {1: 7}])
@pytest.mark.parametrize("container", ["struct", "nested_struct", "list", "array"])
def test_materialized_output_validates_struct_fields_before_encoding(value, container):
    import vane

    dtype = pa.struct([("x", pa.int64())])
    if container == "nested_struct":
        dtype = pa.struct([("child", dtype)])
        value = {"CHILD": value}
    elif container in ("list", "array"):
        dtype = pa.list_(dtype) if container == "list" else pa.list_(dtype, 1)
        value = [value]
    schema = pa.schema([("tensor", pa.fixed_shape_tensor(pa.int64(), [3])), ("record", dtype)])
    with pytest.raises(vane.InvalidInputException, match="exactly the declared fields") as error:
        columns_to_output_table(
            {"tensor": np.arange(3).reshape(1, 3), "record": [value]}, schema, udf_name="field-validation"
        )
    assert "private-extra-value" not in str(error.value)


@pytest.mark.parametrize("container", [deque, UserList])
@pytest.mark.parametrize("fixed", [False, True])
def test_materialized_output_checks_nested_sequence_struct_names(container, fixed):
    dtype = pa.list_(pa.struct([("x", pa.int64())]), 1 if fixed else -1)
    value = container([{"X": 7}])
    output = columns_to_output_table({"record": [value]}, pa.schema([("record", dtype)]), udf_name="sequence")
    assert output.column("record").to_pylist() == [[{"x": 7}]]
    assert value[0] == {"X": 7}


@pytest.mark.parametrize("container", [deque, UserList])
@pytest.mark.parametrize("fixed", [False, True])
@pytest.mark.parametrize("value", [{"X": 7, "extra": "private-extra-value"}, {}, {"x": 7, "X": 8}])
def test_materialized_output_rejects_invalid_fields_in_nested_sequences(container, fixed, value):
    import vane

    dtype = pa.struct([("items", pa.list_(pa.struct([("x", pa.int64())]), 1 if fixed else -1))])
    with pytest.raises(vane.InvalidInputException, match="exactly the declared fields|ambiguous") as error:
        columns_to_output_table(
            {"record": [{"ITEMS": container([value])}]}, pa.schema([("record", dtype)]), udf_name="sequence"
        )
    assert "private-extra-value" not in str(error.value)


def test_materialized_output_rejects_unchecked_nested_containers():
    class UnregisteredSequence:
        def __len__(self):
            return 1

        def __getitem__(self, index):
            if index != 0:
                raise IndexError(index)
            return {"x": 7, "extra": "private-extra-value"}

    dtype = pa.list_(pa.struct([("x", pa.int64())]))
    with pytest.raises(ValueError, match="could not encode") as error:
        columns_to_output_table(
            {"record": [UnregisteredSequence()]}, pa.schema([("record", dtype)]), udf_name="sequence"
        )
    assert "private-extra-value" not in str(error.value)


@pytest.mark.parametrize(
    "values,declared,actual",
    [
        ([1.75, -2.5], pa.int64(), pa.float64()),
        (np.array([1.75, -2.5]), pa.int64(), pa.float64()),
        ([b"\x00A"], pa.string(), pa.binary()),
        ([2**53 + 1], pa.float64(), pa.int64()),
    ],
)
def test_materialized_output_preserves_source_types_for_duckdb_casts(values, declared, actual):
    schema = pa.schema([("tensor", pa.fixed_shape_tensor(pa.int64(), [2])), ("value", declared)])
    tensor = np.zeros((len(values), 2), dtype=np.int64)
    output = columns_to_output_table({"tensor": tensor, "value": values}, schema, udf_name="casts")
    assert output.column("value").type == actual
    assert output.column("value").to_pylist() == list(values)
    assert np.shares_memory(output.column("tensor").chunk(0).to_numpy_ndarray(), tensor)


@pytest.mark.parametrize("backend", ["subprocess_actor", pytest.param("ray_actor", marks=pytest.mark.real_ray)])
def test_actor_public_materialized_casts_match_arrow_output(request, monkeypatch, backend):
    import vane

    if backend == "ray_actor":
        request.getfixturevalue("ray_local")
    monkeypatch.setenv("VANE_RUNNER", "ray" if backend == "ray_actor" else "local-fast")
    outputs = []
    for materialized in (False, True):

        class Output:
            def __call__(self, table):
                ids = table.column("id").to_pylist()
                records = [{"X": [1.75, -2.5][i], "Text": b"\x00A"} for i in ids]
                columns = {
                    "id": ids,
                    "tensor": np.zeros((len(ids), 2), dtype=np.int64),
                    "number": [record["X"] for record in records],
                    "text": [record["Text"] for record in records],
                    "record": records,
                    "items": [deque([record]) for record in records],
                    "fixed": [UserList([record]) for record in records],
                    "empty": [[] for _ in ids],
                }
                if materialized:
                    return columns
                columns["tensor"] = pa.FixedShapeTensorArray.from_numpy_ndarray(columns["tensor"])
                columns["items"] = [list(items) for items in columns["items"]]
                columns["fixed"] = [list(items) for items in columns["fixed"]]
                return pa.table(columns)

        with vane.connect(config={"threads": 2}) as con:
            output = (
                con.sql("select range as id from range(2)")
                .map_batches(
                    Output,
                    schema={
                        "id": "BIGINT",
                        "tensor": vane.tensor_type(vane.sqltypes.BIGINT, [2]),
                        "number": "BIGINT",
                        "text": "VARCHAR",
                        "record": "STRUCT(x BIGINT, text VARCHAR)",
                        "items": "STRUCT(x BIGINT, text VARCHAR)[]",
                        "fixed": "STRUCT(x BIGINT, text VARCHAR)[1]",
                        "empty": "STRUCT(x BIGINT)[]",
                    },
                    batch_size=2,
                    execution_backend=backend,
                    actor_number=1,
                )
                .to_arrow_table()
                .sort_by("id")
            )
        outputs.append(output)
    assert outputs[0].column("number").to_pylist() == [2, -2]
    assert outputs[0].column("text").to_pylist() == ["\\x00A", "\\x00A"]
    assert outputs[0].column("empty").type == pa.list_(pa.struct([("x", pa.int64())]))
    assert outputs[1].equals(outputs[0], check_metadata=True)


@pytest.mark.parametrize("layout", ["middle", "trailing", "row", "offset"])
@pytest.mark.parametrize("dtype", [np.int64, np.float32])
def test_materialized_output_canonicalizes_contiguous_tensor_views(layout, dtype):
    base = np.arange(12, dtype=dtype).reshape(4, 3)
    values = {
        "middle": base[:, None, :],
        "trailing": base[:, :, None],
        "row": base[None, :, :],
        "offset": base[1:3, None, :],
    }[layout]
    values.flags.writeable = False
    assert values.flags.c_contiguous and 0 in values.strides
    expected = values.copy()
    tensor_type = pa.fixed_shape_tensor(pa.from_numpy_dtype(dtype), values.shape[1:])
    output = columns_to_output_table({"tensor": values}, pa.schema([("tensor", tensor_type)]), udf_name="view")
    tensor = output.column("tensor").chunk(0)

    assert tensor.type == tensor_type
    assert tensor.type.permutation is None
    actual = tensor.to_numpy_ndarray()
    assert np.shares_memory(actual, values)
    del output, tensor, values, base
    gc.collect()
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("backend", ["subprocess_actor", pytest.param("ray_actor", marks=pytest.mark.real_ray)])
def test_actor_public_materialized_struct_names_and_singleton_tensor(request, monkeypatch, backend):
    import vane

    if backend == "ray_actor":
        request.getfixturevalue("ray_local")
    monkeypatch.setenv("VANE_RUNNER", "ray" if backend == "ray_actor" else "local-fast")

    class Output:
        def __call__(self, table):
            ids = table.column("id").to_numpy()
            values = ids[:, None] * 3 + np.arange(3, dtype=np.int64)
            return {
                "id": ids,
                "tensor": values[:, None, :],
                "record": [{"X": int(i) + 7, "Items": [{"Y": int(i) + 9}]} for i in ids],
            }

    with vane.connect(config={"threads": 2}) as con:
        output = (
            con.sql("select range as id from range(2)")
            .map_batches(
                Output,
                schema={
                    "id": vane.sqltypes.BIGINT,
                    "tensor": vane.tensor_type(vane.sqltypes.BIGINT, [1, 3]),
                    "record": vane.type("STRUCT(x BIGINT, items STRUCT(y BIGINT)[])"),
                },
                batch_size=2,
                execution_backend=backend,
                actor_number=1,
            )
            .to_arrow_table()
        )
    output = output.sort_by("id")
    assert output.column("record").to_pylist() == [
        {"x": 7, "items": [{"y": 9}]},
        {"x": 8, "items": [{"y": 10}]},
    ]
    np.testing.assert_array_equal(
        output.column("tensor").combine_chunks().to_numpy_ndarray(), np.arange(6).reshape(2, 1, 3)
    )


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
        columns_to_output_table({"y": [1, "private-pixel-content"]}, pa.schema([("y", pa.int64())]), udf_name="test")
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
@pytest.mark.parametrize("batch_format", ["pyarrow", "numpy"])
def test_numpy_tensor_output_without_pandas(rows, batch_format):
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
                return {
                    "pixels": np.arange(rows * 6, dtype=np.uint8).reshape(rows, 2, 3),
                    "value": np.full(rows, 1.75) if sys.argv[2] == "numpy" else [1.75] * rows,
                }

        runtime = UDFExecutor({
            "function_pickle": dumps(Output),
            "call_mode": "map_batches",
            "execution_backend": "subprocess_actor",
            "batch_format": sys.argv[2],
            "output_schema": [
                {"name": "pixels", "kind": "tensor", "dtype": "UTINYINT", "shape": [2, 3]},
                {"name": "value", "type": "BIGINT"},
            ],
        })
        try:
            result = list(runtime.iter_submit(pa.table({"x": [1]})))[0]
            assert result.num_rows == rows
            assert result.column(0).type == pa.fixed_shape_tensor(pa.uint8(), [2, 3])
            assert result.column("value").to_pylist() == [1.75] * rows
            assert result.column("value").type == (pa.float64() if rows else pa.int64())
            if rows:
                np.testing.assert_array_equal(
                    result.column(0).chunk(0).to_numpy_ndarray(), np.arange(6, dtype=np.uint8).reshape(1, 2, 3)
                )
            assert "pandas" not in sys.modules
        finally:
            runtime.close()
    """)
    completed = subprocess.run(
        [sys.executable, "-I", "-c", script, str(rows), batch_format], capture_output=True, text=True, timeout=30
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
