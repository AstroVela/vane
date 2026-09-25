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
    pixels = np.arange(3 * 640 * 640 * 3, dtype=np.uint8).reshape(3, 640, 640, 3)
    frames = _frames(kind, pixels)
    for column in (frames.slice(1, 2), pa.chunked_array([frames.slice(1, 1), frames.slice(2, 1)])):
        result = benchmark._frame_batch(column)
        np.testing.assert_array_equal(result, pixels[1:])
        assert result.dtype == np.uint8 and result.flags.c_contiguous
    assert benchmark._frame_batch(frames.slice(0, 0)).shape == (0, 640, 640, 3)
    with pytest.raises(ValueError, match="NULL"):
        benchmark._frame_batch(frames.take(pa.array([None], type=pa.int64())))


@pytest.mark.parametrize("profile_enabled", [False, True])
def test_python_video_detector_keeps_tensor_output_and_pillow_crops(benchmark, monkeypatch, tmp_path, profile_enabled):
    pil = pytest.importorskip("PIL.Image")
    pixels = np.arange(640 * 640 * 3, dtype=np.uint8).reshape(1, 640, 640, 3)
    frame = _frames("generic", pixels)
    features = [dict(label=2, confidence=0.9, bbox=[-1.9, 1.9, 3.9, 5.9])]
    captured = []

    def predict(tensor, *, verbose):
        captured.append(tensor.numpy())
        return [SimpleNamespace()]

    monkeypatch.setattr(benchmark, "yolo_result_to_features", lambda result: features)
    monkeypatch.setattr(
        benchmark, "predict_with_original_frames", lambda model, tensor, frames: model(tensor, verbose=False)
    )
    detector = object.__new__(benchmark.YOLODetector)
    detector.model = predict
    detector.profile = benchmark.BatchProfile(str(tmp_path), "vane") if profile_enabled else None
    detected = detector(pa.table({"frame_index": [7], "frame": frame}))
    assert isinstance(detected["frame"].combine_chunks(), pa.FixedShapeTensorArray)
    np.testing.assert_array_equal(benchmark._frame_batch(detected["frame"]), pixels)
    np.testing.assert_allclose(captured[0], pixels.transpose(0, 3, 1, 2).astype(np.float32) / 255)
    cropped = benchmark._crop_objects(detected)
    assert cropped["frame_index"].to_pylist() == [7]
    assert cropped["features"].to_pylist() == features
    with pil.open(io.BytesIO(cropped["object"][0].as_py())) as actual:
        expected = pil.fromarray(pixels[0]).crop((-1, 1, 3, 5))
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


@pytest.mark.parametrize("rows", [1, 3])
def test_vane_batched_tensor_matches_reference(benchmark, rows):
    import torch
    from video_kernels import frames_to_torch_tensor

    # Include every uint8 value and non-contiguous input, preserving RGB order.
    pixels = np.arange(rows * 32 * 32 * 3, dtype=np.uint8).reshape(rows, 32, 32, 3)[:, ::2]
    pixels.setflags(write=False)
    actual = benchmark.batch_to_tensor(pixels)
    expected = frames_to_torch_tensor(pixels, None)
    assert actual.is_contiguous()
    assert torch.equal(actual, expected)


def test_vane_predictor_reuses_originals_and_validates_shape(benchmark, monkeypatch):
    import torch
    from vane_video_kernels import OriginalFramePredictor

    predictor = object.__new__(OriginalFramePredictor)
    frames = np.zeros((2, 16, 16, 3), dtype=np.uint8)
    predictor.original_frames = frames
    parent = OriginalFramePredictor.__bases__[0]
    monkeypatch.setattr(parent, "postprocess", lambda self, preds, img, orig_imgs, **kw: orig_imgs)
    result = predictor.postprocess(None, torch.zeros((2, 3, 16, 16)), None)
    assert isinstance(result, list) and len(result) == 2
    assert np.shares_memory(result[0], frames)
    with pytest.raises(ValueError, match="match the current"):
        predictor.postprocess(None, torch.zeros((1, 3, 16, 16)), None)
    with pytest.raises(ValueError, match="already-resized"):
        predictor.postprocess(None, torch.zeros((2, 3, 32, 32)), None)


@pytest.mark.parametrize("fail", [False, True])
def test_vane_predictor_releases_frames_after_each_call(benchmark, fail):
    from vane_video_kernels import OriginalFramePredictor, predict_with_original_frames

    class Model:
        predictor = None

        def __call__(self, tensor, *, verbose, predictor):
            if self.predictor is None:
                self.predictor = object.__new__(OriginalFramePredictor)
                self.predictor.original_frames = predictor.keywords["original_frames"]
            assert self.predictor.original_frames is frames
            if fail:
                raise RuntimeError("inference failed")
            return "results"

    model = Model()
    for _ in range(2):
        frames = np.zeros((1, 16, 16, 3), dtype=np.uint8)
        if fail:
            with pytest.raises(RuntimeError, match="inference failed"):
                predict_with_original_frames(model, None, frames)
        else:
            assert predict_with_original_frames(model, None, frames) == "results"
        assert model.predictor.original_frames is None


@pytest.mark.parametrize("profile_enabled", [False, True])
@pytest.mark.parametrize("rows", [0, 2])
def test_ray_video_detector_keeps_results_with_profile(benchmark, monkeypatch, tmp_path, profile_enabled, rows):
    pytest.importorskip("ray")
    spec = importlib.util.spec_from_file_location(
        "ray_video_benchmark", Path(benchmark.__file__).with_name("ray_data_main.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    features = [dict(label=2, confidence=0.9, bbox=[1, 2, 3, 4])]
    monkeypatch.setattr(module, "yolo_result_to_features", lambda result: features)
    detector = object.__new__(module.ExtractImageFeatures)
    detector.model = lambda tensor, **kwargs: [SimpleNamespace()] * len(tensor)
    detector.profile = module.BatchProfile(str(tmp_path), "ray_data") if profile_enabled else None
    pixels = np.full((rows, 640, 640, 3), 42, dtype=np.uint8)
    batch = dict(frame=pixels)
    result = detector(batch)
    assert result is batch
    assert result["frame"] is pixels
    assert result["features"] == [features] * rows


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
    pixels = benchmark._frame_batch(table["frame"])
    for frame, level in zip(pixels, (17, 91, 203), strict=True):
        np.testing.assert_array_equal(frame, np.full((640, 640, 3), level, dtype=np.uint8))
