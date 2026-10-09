# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import io
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pytest

import vane
from vane._image import image_arrow_type
from vane.execution.udf_batch_format import format_udf_input, iter_udf_output_tables


def _output_table(batch, schema):
    fields = []
    for name, dtype in schema.items():
        if dtype.id == "tensor":
            children = dict(dtype.children)
            fields.append({"name": name, "kind": "tensor", "dtype": str(children["dtype"]), "shape": children["shape"]})
        else:
            fields.append({"name": name, "kind": "duckdb_type", "type": str(dtype)})
    return next(iter_udf_output_tables(batch, batch_format="numpy", output_schema=fields))


@pytest.fixture
def benchmark(monkeypatch):
    pytest.importorskip("ultralytics")
    directory = Path(__file__).resolve().parents[2] / "multimodal_inference_benchmarks/video_object_detection"
    monkeypatch.syspath_prepend(str(directory))
    spec = importlib.util.spec_from_file_location("video_benchmark_python", directory / "vane_main.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _frames(kind, pixels):
    if kind == "tensor":
        return pa.FixedShapeTensorArray.from_numpy_ndarray(pixels)
    dtype = vane.image_type() if kind == "generic" else vane.image_type("RGB", 640, 640)
    arrow_type = image_arrow_type(dtype)
    if kind == "generic":
        values = [dict(data=p.reshape(-1).astype(np.float32), channel=3, height=640, width=640, mode=3) for p in pixels]
    else:
        values = [p.reshape(-1) for p in pixels]
    return pa.ExtensionArray.from_storage(arrow_type, pa.array(values, type=arrow_type.storage_type))


@pytest.mark.parametrize("kind", ["tensor", "generic", "fixed"])
def test_python_video_frame_conversion_preserves_pixels(benchmark, kind):
    pixels = np.stack([np.full((640, 640, 3), level, dtype=np.uint8) for level in (17, 91, 203)])
    frames = _frames(kind, pixels)
    for column in (
        frames.slice(1, 2),
        pa.chunked_array([frames.slice(1, 2)]),
        pa.chunked_array([frames.slice(1, 1), frames.slice(2, 1)]),
    ):
        batch = format_udf_input(pa.table({"frame": column}), "numpy")
        result = benchmark._frame_batch(batch["frame"])
        assert result is batch["frame"]
        np.testing.assert_array_equal(np.stack(result), pixels[1:])
        assert all(frame.dtype == np.uint8 and not frame.flags.writeable for frame in result)
        if kind == "tensor" and isinstance(column, pa.Array):
            assert np.shares_memory(result, pixels)
    empty = format_udf_input(pa.table({"frame": frames.slice(0, 0)}), "numpy")["frame"]
    assert len(benchmark._frame_batch(empty)) == 0
    with pytest.raises(ValueError, match="NULL"):
        nulls = frames.take(pa.array([None], type=pa.int64()))
        benchmark._frame_batch(format_udf_input(pa.table({"frame": nulls}), "numpy")["frame"])


@pytest.mark.parametrize("kind", ["tensor", "generic", "fixed"])
@pytest.mark.parametrize("detections", [0, 1, 2])
def test_python_video_detector_keeps_tensor_output_and_pillow_crops(benchmark, monkeypatch, kind, detections):
    pil = pytest.importorskip("PIL.Image")
    pixels = np.arange(640 * 640 * 3, dtype=np.uint8).reshape(1, 640, 640, 3)
    frame = _frames(kind, pixels)
    features = [dict(label=2, confidence=0.9, bbox=[-1.9, 1.9, 3.9, 5.9])] * detections
    captured = []

    def predict(tensor, *, verbose):
        captured.append(tensor.numpy())
        return [SimpleNamespace()]

    monkeypatch.setattr(benchmark, "yolo_result_to_features", lambda result: features)
    detector = object.__new__(benchmark.YOLODetector)
    detector.model = predict
    batch = format_udf_input(pa.table({"frame_index": [7], "frame": frame}), "numpy")
    detected_batch = detector(batch)
    assert detected_batch["frame"] is batch["frame"]
    detected = _output_table(
        detected_batch,
        {"frame_index": vane.sqltypes.BIGINT, "frame": benchmark.FRAME_TYPE, "features": benchmark.FEATURE_LIST_TYPE},
    )
    tensor = detected["frame"].chunk(0)
    assert isinstance(tensor, pa.FixedShapeTensorArray)
    np.testing.assert_array_equal(tensor.to_numpy_ndarray(), pixels)
    if kind != "generic":
        assert np.shares_memory(tensor.to_numpy_ndarray(), batch["frame"])
    np.testing.assert_allclose(captured[0], pixels.transpose(0, 3, 1, 2).astype(np.float32) / 255)
    cropped = _output_table(
        benchmark._crop_objects(format_udf_input(detected, "numpy")),
        {"frame_index": vane.sqltypes.BIGINT, "features": benchmark.FEATURE_TYPE, "object": vane.sqltypes.BLOB},
    )
    assert cropped["frame_index"].to_pylist() == [7] * detections
    assert cropped["features"].to_pylist() == features
    for value in cropped["object"].to_pylist():
        with pil.open(io.BytesIO(value)) as actual:
            expected = pil.fromarray(pixels[0]).crop((-1, 1, 3, 5))
            np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


@pytest.mark.parametrize(
    "shape,dtype", [((1, 640, 640, 1), np.uint8), ((1, 320, 640, 3), np.uint8), ((1, 640, 640, 3), np.float32)]
)
def test_python_video_keeps_model_frame_validation(benchmark, shape, dtype):
    with pytest.raises(ValueError, match="Unexpected frame"):
        benchmark._frame_batch(np.zeros(shape, dtype=dtype))


def test_python_video_crop_skips_null_and_empty_detections(benchmark):
    pixels = np.zeros((3, 640, 640, 3), dtype=np.uint8)
    feature = {"label": 2, "confidence": 0.9, "bbox": [0.0, 0.0, 2.0, 2.0]}
    features = pa.array(
        [None, [], [feature]],
        type=pa.list_(
            pa.struct([("label", pa.int64()), ("confidence", pa.float64()), ("bbox", pa.list_(pa.float64()))])
        ),
    )
    table = pa.table({"frame_index": [1, 2, 3], "frame": _frames("tensor", pixels), "features": features})
    output = benchmark._crop_objects(format_udf_input(table, "numpy"))
    assert output["frame_index"].tolist() == [3]
    assert output["features"].tolist()[0]["label"] == 2


def test_python_video_source_exposes_fixed_tensor_frames(benchmark, tmp_path, monkeypatch):
    av = pytest.importorskip("av")
    monkeypatch.setenv("VANE_RUNNER", "local")
    path = tmp_path / "frames.avi"
    with av.open(str(path), "w") as container:
        stream = container.add_stream("ffv1", rate=4)
        stream.width, stream.height, stream.pix_fmt = 16, 12, "bgr0"
        for level in (17, 91, 203):
            frame = av.VideoFrame.from_ndarray(np.full((12, 16, 3), level, dtype=np.uint8), format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    with vane.connect() as con:
        source = benchmark.PythonVideoFrameSource([str(path)], height=640, width=640)
        relation = benchmark.read_datasource(source, con=con).project("frame_index, frame")
        table = relation.arrow().read_all()
    assert table["frame_index"].to_pylist() == [0, 1, 2]
    assert isinstance(table["frame"].combine_chunks(), pa.FixedShapeTensorArray)
    pixels = benchmark._frame_batch(format_udf_input(table, "numpy")["frame"])
    for frame, level in zip(pixels, (17, 91, 203), strict=True):
        np.testing.assert_array_equal(frame, np.full((640, 640, 3), level, dtype=np.uint8))
