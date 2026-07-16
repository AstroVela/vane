from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_VIDEO_DIR = _REPO_ROOT / "multimodal_inference_benchmarks/video_object_detection"
_VIDEO_VANE = _VIDEO_DIR / "vane_main.py"


def _timing_tokens(text: str) -> dict[str, str]:
    tokens: dict[str, str] = {}
    for item in text.split():
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        tokens[key] = value
    return tokens


def _load_vane_main():
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    pytest.importorskip("PIL")
    pytest.importorskip("ultralytics")

    module_name = "video_vane_main_timing_test"
    sys.modules.pop(module_name, None)
    if str(_VIDEO_DIR) not in sys.path:
        sys.path.insert(0, str(_VIDEO_DIR))
    spec = importlib.util.spec_from_file_location(module_name, _VIDEO_VANE)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_video_udf_timing_fields_include_ms_aliases():
    vane_main = _load_vane_main()

    fields = vane_main._format_video_timing_fields(
        frame_index_s=0.001,
        frame_s=0.002,
        tensor_s=0.0035,
        model_s=0.004,
        feature_s=0.005,
        arrow_s=0.006,
        total_s=0.020,
        rows_per_s=123.456,
    )

    tokens = _timing_tokens(fields)
    assert tokens["frame_index_s"] == "0.001000"
    assert tokens["frame_index_ms"] == "1.000"
    assert tokens["tensor_s"] == "0.003500"
    assert tokens["tensor_ms"] == "3.500"
    assert tokens["model_ms"] == "4.000"
    assert tokens["total_ms"] == "20.000"
    assert tokens["rows_per_s"] == "123.46"


def test_video_frame_tensor_matches_torchvision_reference(monkeypatch):
    torch = pytest.importorskip("torch")
    import torchvision
    from PIL import Image

    vane_main = _load_vane_main()
    monkeypatch.setattr(vane_main, "FRAME_HEIGHT", 4)
    monkeypatch.setattr(vane_main, "FRAME_WIDTH", 5)
    frames = np.arange(2 * 4 * 5 * 3, dtype=np.uint8).reshape(2, 4, 5, 3)
    frame_col = pa.FixedShapeTensorArray.from_numpy_ndarray(frames)

    actual = vane_main._frame_column_to_tensor_batch(frame_col, torch.device("cpu"))
    expected = torch.stack(
        [torchvision.transforms.functional.to_tensor(Image.fromarray(frame)) for frame in frames],
        dim=0,
    )

    torch.testing.assert_close(actual, expected)


def test_video_frame_tensor_single_chunked_array_reuses_existing_chunk(monkeypatch):
    vane_main = _load_vane_main()
    monkeypatch.setattr(vane_main, "FRAME_HEIGHT", 1)
    monkeypatch.setattr(vane_main, "FRAME_WIDTH", 2)
    frames = np.arange(1 * 1 * 2 * 3, dtype=np.uint8).reshape(1, 1, 2, 3)
    chunk = pa.FixedShapeTensorArray.from_numpy_ndarray(frames)
    chunked = pa.chunked_array([chunk])

    normalized = vane_main._normalize_frame_column(chunked)
    frame_batch = vane_main._frame_column_to_frame_batch(chunked)

    assert normalized.storage.buffers()[2].address == chunk.storage.buffers()[2].address
    np.testing.assert_array_equal(frame_batch, frames)


def test_video_udf_timing_line_appends_to_configured_file(tmp_path, monkeypatch, capsys):
    vane_main = _load_vane_main()
    log_path = tmp_path / "timing" / "video_udf.log"
    monkeypatch.setattr(vane_main, "UDF_TIMING_LOG_PATH", str(log_path))

    vane_main._emit_video_timing_line("[vane_video][crop_udf_timing] run_id=test rows=1 total_ms=2.000")

    captured = capsys.readouterr()
    assert "total_ms=2.000" in captured.err
    assert log_path.read_text(encoding="utf-8").strip().endswith("total_ms=2.000")


def test_video_arrow_buffer_helpers_preserve_values():
    vane_main = _load_vane_main()

    features = [
        {"label": 1, "confidence": 0.5, "bbox": [1.0, 2.0, 3.0, 4.0]},
        {"label": 2, "confidence": 0.25, "bbox": [5.0, 6.0, 7.0, 8.0]},
    ]

    assert vane_main._int64_array([1, 2, 3]).to_pylist() == [1, 2, 3]
    assert vane_main._binary_array([b"abc", b"", b"de"]).to_pylist() == [b"abc", b"", b"de"]
    feature_array = vane_main._feature_array(features)
    assert feature_array.type == vane_main.FEATURE_ARROW_TYPE
    feature_values = feature_array.to_pylist()
    assert [item["label"] for item in feature_values] == [1, 2]
    assert [vane_main._feature_field(item, "label") for item in feature_values] == [1, 2]
    assert [vane_main._feature_field(item, "confidence") for item in feature_values] == [0.5, 0.25]
    assert [vane_main._feature_field(item, "bbox") for item in feature_values] == [
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
    ]
    assert feature_array.type.field("confidence").type == pa.float64()
    assert feature_array.type.field("bbox").type.value_type == pa.float64()
    nested_values = vane_main._features_array([[features[0]], [], [features[1]]]).to_pylist()
    assert [vane_main._feature_field(item, "label") for item in nested_values[0]] == [1]
    assert nested_values[1] == []
    assert [vane_main._feature_field(item, "label") for item in nested_values[2]] == [2]
    assert vane_main._feature_field({"label": None, '"label"': 7}, "label") == 7


def test_video_frame_tensor_handles_multi_chunk_with_torchvision_reference(monkeypatch):
    torch = pytest.importorskip("torch")
    import torchvision
    from PIL import Image

    vane_main = _load_vane_main()
    monkeypatch.setattr(vane_main, "FRAME_HEIGHT", 2)
    monkeypatch.setattr(vane_main, "FRAME_WIDTH", 3)
    frames = np.arange(3 * 2 * 3 * 3, dtype=np.uint8).reshape(3, 2, 3, 3)
    chunked = pa.chunked_array(
        [
            pa.FixedShapeTensorArray.from_numpy_ndarray(frames[:1]),
            pa.FixedShapeTensorArray.from_numpy_ndarray(frames[1:]),
        ]
    )

    actual = vane_main._frame_column_to_tensor_batch(chunked, torch.device("cpu"))
    expected = torch.stack(
        [torchvision.transforms.functional.to_tensor(Image.fromarray(frame)) for frame in frames],
        dim=0,
    )

    torch.testing.assert_close(actual, expected)


def test_video_frame_views_do_not_combine_multi_chunk_tensor(monkeypatch):
    vane_main = _load_vane_main()
    monkeypatch.setattr(vane_main, "FRAME_HEIGHT", 2)
    monkeypatch.setattr(vane_main, "FRAME_WIDTH", 3)
    frames = np.arange(3 * 2 * 3 * 3, dtype=np.uint8).reshape(3, 2, 3, 3)
    first_chunk = pa.FixedShapeTensorArray.from_numpy_ndarray(frames[:1])
    second_chunk = pa.FixedShapeTensorArray.from_numpy_ndarray(frames[1:])
    chunked = pa.chunked_array([first_chunk, second_chunk])

    frame_views = vane_main._frame_column_to_frames(chunked)

    assert len(frame_views) == 3
    assert np.shares_memory(frame_views[0], first_chunk.to_numpy_ndarray())
    assert np.shares_memory(frame_views[1], second_chunk.to_numpy_ndarray())
    np.testing.assert_array_equal(np.stack(frame_views), frames)


def test_video_crop_udf_timing_logs_when_enabled(monkeypatch, capsys):
    vane_main = _load_vane_main()
    monkeypatch.setattr(vane_main, "CPU_UDF_TIMING_LOG", True)
    monkeypatch.setattr(vane_main, "CPU_UDF_TIMING_SAMPLE_RATE", 1)
    monkeypatch.setattr(vane_main, "_CPU_UDF_TIMING_CALLS", 0)
    monkeypatch.setattr(vane_main, "BENCHMARK_RUN_ID", "unit")
    monkeypatch.setattr(vane_main, "UDF_TIMING_LOG_PATH", None)

    frame = np.zeros((vane_main.FRAME_HEIGHT, vane_main.FRAME_WIDTH, 3), dtype=np.uint8)
    features = [[{"label": 0, "confidence": 0.5, "bbox": [0.0, 0.0, 1.0, 1.0]}]]
    table = pa.table(
        {
            "frame_index": pa.array([0], type=pa.int64()),
            "frame": pa.FixedShapeTensorArray.from_numpy_ndarray(frame.reshape(1, *frame.shape)),
            "features": vane_main._features_array(features),
        }
    )

    result = next(vane_main._crop_generator(table))

    captured = capsys.readouterr()
    assert result.num_rows == 1
    assert result.column_names == ["frame_index", "features", "object"]
    assert "[vane_video][crop_udf_timing]" in captured.err
    assert "run_id=unit" in captured.err
    assert "rows=1" in captured.err
    assert "output_rows=1" in captured.err
    assert "crop_encode_ms=" in captured.err
    assert "crop_pil_ms=" in captured.err
    assert "png_encode_ms=" in captured.err
    assert "bbox_area=" in captured.err
    assert "png_bytes=" in captured.err
    assert "frame_index_array_ms=" in captured.err
    assert "feature_array_ms=" in captured.err
    assert "object_array_ms=" in captured.err
    assert "table_build_ms=" in captured.err
    assert "write_ms=" not in captured.err


def test_video_crop_udf_reuses_png_buffer(monkeypatch):
    vane_main = _load_vane_main()
    monkeypatch.setattr(vane_main, "CPU_UDF_TIMING_LOG", False)
    monkeypatch.setattr(vane_main, "_CPU_UDF_TIMING_CALLS", 0)
    calls = 0
    real_bytes_io = vane_main.io.BytesIO

    def counted_bytes_io(*args, **kwargs):
        nonlocal calls
        calls += 1
        return real_bytes_io(*args, **kwargs)

    monkeypatch.setattr(vane_main.io, "BytesIO", counted_bytes_io)
    frame = np.zeros((vane_main.FRAME_HEIGHT, vane_main.FRAME_WIDTH, 3), dtype=np.uint8)
    features = [
        [
            {"label": 0, "confidence": 0.5, "bbox": [0.0, 0.0, 1.0, 1.0]},
            {"label": 1, "confidence": 0.6, "bbox": [1.0, 1.0, 2.0, 2.0]},
        ]
    ]
    table = pa.table(
        {
            "frame_index": pa.array([0], type=pa.int64()),
            "frame": pa.FixedShapeTensorArray.from_numpy_ndarray(frame.reshape(1, *frame.shape)),
            "features": vane_main._features_array(features),
        }
    )

    result = next(vane_main._crop_generator(table))

    assert result.num_rows == 2
    assert result.column_names == ["frame_index", "features", "object"]
    assert all(isinstance(value, bytes) for value in result.column("object").to_pylist())
    assert calls == 1


def test_video_crop_udf_returns_ray_data_shaped_rows(monkeypatch):
    vane_main = _load_vane_main()
    monkeypatch.setattr(vane_main, "CPU_UDF_TIMING_LOG", False)
    monkeypatch.setattr(vane_main, "_CPU_UDF_TIMING_CALLS", 0)

    frame = np.zeros((vane_main.FRAME_HEIGHT, vane_main.FRAME_WIDTH, 3), dtype=np.uint8)
    features = [[{"label": 7, "confidence": 0.75, "bbox": [0.0, 0.0, 2.0, 2.0]}]]
    table = pa.table(
        {
            "frame_index": pa.array([42], type=pa.int64()),
            "frame": pa.FixedShapeTensorArray.from_numpy_ndarray(frame.reshape(1, *frame.shape)),
            "features": vane_main._features_array(features),
        }
    )

    result = next(vane_main._crop_generator(table))

    assert result.column_names == ["frame_index", "features", "object"]
    assert result.column("frame_index").to_pylist() == [42]
    assert vane_main._feature_field(result.column("features").to_pylist()[0], "label") == 7
    assert vane_main._feature_field(result.column("features").to_pylist()[0], "bbox") == [0.0, 0.0, 2.0, 2.0]
    assert isinstance(result.column("object").to_pylist()[0], bytes)


def test_video_crop_flat_map_returns_ray_data_shaped_rows(monkeypatch):
    vane_main = _load_vane_main()
    monkeypatch.setattr(vane_main, "CPU_UDF_TIMING_LOG", False)
    monkeypatch.setattr(vane_main, "_CPU_UDF_TIMING_CALLS", 0)

    frame = np.zeros((vane_main.FRAME_HEIGHT, vane_main.FRAME_WIDTH, 3), dtype=np.uint8)
    feature = {"label": 7, "confidence": 0.75, "bbox": [0.0, 0.0, 2.0, 2.0]}
    row = {
        "frame_index": 42,
        "frame": frame,
        "features": [feature],
    }

    rows = list(vane_main._crop_flat_map(row))

    assert len(rows) == 1
    assert rows[0]["frame_index"] == 42
    assert rows[0]["features"] == feature
    assert isinstance(rows[0]["object"], bytes)


def test_video_crop_flat_map_accepts_flat_fixed_shape_tensor_scalar(monkeypatch):
    vane_main = _load_vane_main()
    monkeypatch.setattr(vane_main, "CPU_UDF_TIMING_LOG", False)
    monkeypatch.setattr(vane_main, "_CPU_UDF_TIMING_CALLS", 0)
    monkeypatch.setattr(vane_main, "FRAME_HEIGHT", 2)
    monkeypatch.setattr(vane_main, "FRAME_WIDTH", 3)

    frame = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
    feature = {"label": 7, "confidence": 0.75, "bbox": [0.0, 0.0, 2.0, 2.0]}
    row = {
        "frame_index": 42,
        "frame": frame.reshape(-1),
        "features": [feature],
    }

    rows = list(vane_main._crop_flat_map(row))

    assert len(rows) == 1
    assert rows[0]["frame_index"] == 42
    assert rows[0]["features"] == feature
    assert isinstance(rows[0]["object"], bytes)


def test_video_crop_flat_map_timing_logs_when_enabled(monkeypatch, capsys):
    vane_main = _load_vane_main()
    monkeypatch.setattr(vane_main, "CPU_UDF_TIMING_LOG", True)
    monkeypatch.setattr(vane_main, "CPU_UDF_TIMING_SAMPLE_RATE", 1)
    monkeypatch.setattr(vane_main, "_CPU_UDF_TIMING_CALLS", 0)
    monkeypatch.setattr(vane_main, "BENCHMARK_RUN_ID", "unit")
    monkeypatch.setattr(vane_main, "UDF_TIMING_LOG_PATH", None)

    frame = np.zeros((vane_main.FRAME_HEIGHT, vane_main.FRAME_WIDTH, 3), dtype=np.uint8)
    row = {
        "frame_index": 42,
        "frame": frame,
        "features": [{"label": 7, "confidence": 0.75, "bbox": [0.0, 0.0, 2.0, 2.0]}],
    }

    rows = list(vane_main._crop_flat_map(row))

    captured = capsys.readouterr()
    assert len(rows) == 1
    assert "[vane_video][crop_flat_map_timing]" in captured.err
    assert "run_id=unit" in captured.err
    assert "rows=1" in captured.err
    assert "output_rows=1" in captured.err
    assert "crop_encode_ms=" in captured.err
    assert "crop_pil_ms=" in captured.err
    assert "png_encode_ms=" in captured.err
    assert "bbox_area=" in captured.err
    assert "png_bytes=" in captured.err


def test_video_crop_exploded_udf_returns_ray_data_shaped_rows(monkeypatch):
    vane_main = _load_vane_main()
    monkeypatch.setattr(vane_main, "CPU_UDF_TIMING_LOG", False)
    monkeypatch.setattr(vane_main, "_CPU_UDF_TIMING_CALLS", 0)

    frame = np.zeros((vane_main.FRAME_HEIGHT, vane_main.FRAME_WIDTH, 3), dtype=np.uint8)
    feature = {"label": 7, "confidence": 0.75, "bbox": [0.0, 0.0, 2.0, 2.0]}
    table = pa.table(
        {
            "frame_index": pa.array([42], type=pa.int64()),
            "frame": pa.FixedShapeTensorArray.from_numpy_ndarray(frame.reshape(1, *frame.shape)),
            "features": vane_main._feature_array([feature]),
        }
    )

    result = next(vane_main._crop_exploded_generator(table))

    assert result.column_names == ["frame_index", "features", "object"]
    assert result.column("frame_index").to_pylist() == [42]
    assert vane_main._feature_field(result.column("features").to_pylist()[0], "label") == 7
    assert vane_main._feature_field(result.column("features").to_pylist()[0], "bbox") == [0.0, 0.0, 2.0, 2.0]
    assert isinstance(result.column("object").to_pylist()[0], bytes)


def test_video_crop_exploded_udf_skips_null_explode_rows(monkeypatch):
    vane_main = _load_vane_main()
    monkeypatch.setattr(vane_main, "CPU_UDF_TIMING_LOG", False)
    monkeypatch.setattr(vane_main, "_CPU_UDF_TIMING_CALLS", 0)

    frame = np.zeros((vane_main.FRAME_HEIGHT, vane_main.FRAME_WIDTH, 3), dtype=np.uint8)
    feature = {"label": 7, "confidence": 0.75, "bbox": [0.0, 0.0, 2.0, 2.0]}
    table = pa.table(
        {
            "frame_index": pa.array([41, 42], type=pa.int64()),
            "frame": pa.FixedShapeTensorArray.from_numpy_ndarray(np.stack([frame, frame], axis=0)),
            "features": pa.array([None, feature], type=vane_main.FEATURE_ARROW_TYPE),
        }
    )

    result = next(vane_main._crop_exploded_generator(table))

    assert result.num_rows == 1
    assert result.column("frame_index").to_pylist() == [42]
    assert vane_main._feature_field(result.column("features").to_pylist()[0], "label") == 7
    assert vane_main._feature_field(result.column("features").to_pylist()[0], "bbox") == [0.0, 0.0, 2.0, 2.0]


def test_video_crop_exploded_udf_timing_logs_when_enabled(monkeypatch, capsys):
    vane_main = _load_vane_main()
    monkeypatch.setattr(vane_main, "CPU_UDF_TIMING_LOG", True)
    monkeypatch.setattr(vane_main, "CPU_UDF_TIMING_SAMPLE_RATE", 1)
    monkeypatch.setattr(vane_main, "_CPU_UDF_TIMING_CALLS", 0)
    monkeypatch.setattr(vane_main, "BENCHMARK_RUN_ID", "unit")
    monkeypatch.setattr(vane_main, "UDF_TIMING_LOG_PATH", None)

    frame = np.zeros((vane_main.FRAME_HEIGHT, vane_main.FRAME_WIDTH, 3), dtype=np.uint8)
    feature = {"label": 7, "confidence": 0.75, "bbox": [0.0, 0.0, 2.0, 2.0]}
    table = pa.table(
        {
            "frame_index": pa.array([42], type=pa.int64()),
            "frame": pa.FixedShapeTensorArray.from_numpy_ndarray(frame.reshape(1, *frame.shape)),
            "features": vane_main._feature_array([feature]),
        }
    )

    result = next(vane_main._crop_exploded_generator(table))

    captured = capsys.readouterr()
    assert result.num_rows == 1
    assert "[vane_video][crop_exploded_udf_timing]" in captured.err
    assert "run_id=unit" in captured.err
    assert "rows=1" in captured.err
    assert "output_rows=1" in captured.err
    assert "crop_encode_ms=" in captured.err
    assert "crop_pil_ms=" in captured.err
    assert "png_encode_ms=" in captured.err
    assert "bbox_area=" in captured.err
    assert "png_bytes=" in captured.err


def test_video_defaults_use_large_reader_and_backpressure_windows(monkeypatch):
    for name in (
        "VANE_MAX_CONCURRENT_DECODES",
        "VANE_RAY_MAX_TASK_BACKLOG",
        "VANE_RAY_SCAN_TASK_MIN_PARTITION_NUM",
        "VANE_FTE_DYNAMIC_SCAN_MAX_SPLITS_PER_PARTITION",
    ):
        monkeypatch.delenv(name, raising=False)

    vane_main = _load_vane_main()

    assert vane_main.VIDEO_MAX_CONCURRENT_DECODES == 256
    assert vane_main.VIDEO_SCAN_TASK_BACKLOG == 2048
    assert vane_main.VIDEO_SCAN_TASK_MIN_PARTITIONS == 256
    assert vane_main.VIDEO_FTE_DYNAMIC_SCAN_MAX_SPLITS_PER_PARTITION == 1
    assert os.environ["VANE_RAY_SCAN_TASK_MIN_PARTITION_NUM"] == "256"
    assert os.environ["VANE_FTE_DYNAMIC_SCAN_MAX_SPLITS_PER_PARTITION"] == "1"


def test_video_cpu_batch_size_reads_env(monkeypatch):
    monkeypatch.setenv("VIDEO_CPU_BATCH_SIZE", "64")

    vane_main = _load_vane_main()

    assert vane_main.VIDEO_CPU_BATCH_SIZE == 64


def test_video_crop_mode_reads_env(monkeypatch):
    monkeypatch.setenv("VIDEO_CROP_MODE", "flat_map")

    vane_main = _load_vane_main()

    assert vane_main.VIDEO_CROP_MODE == "flat_map"


def test_video_cpu_output_batch_size_reads_env(monkeypatch):
    monkeypatch.setenv("VIDEO_CPU_OUTPUT_BATCH_SIZE", "256")

    vane_main = _load_vane_main()

    assert vane_main.VIDEO_CPU_OUTPUT_BATCH_SIZE == 256


def test_video_cpu_cpus_reads_env(monkeypatch):
    monkeypatch.setenv("VIDEO_CPU_CPUS", "0.5")

    vane_main = _load_vane_main()

    assert vane_main.VIDEO_CPU_CPUS == 0.5
    assert vane_main._cpu_udf_kwargs()["cpus"] == 0.5


def test_video_cpu_memory_bytes_are_stage_specific(monkeypatch):
    monkeypatch.setenv("VIDEO_CPU_MEMORY_BYTES", "268435456")

    vane_main = _load_vane_main()
    monkeypatch.setattr(vane_main, "CPU_UDF_USE_RAY", True)

    assert vane_main.VIDEO_CPU_MEMORY_BYTES == 268435456
    assert vane_main._cpu_udf_kwargs()["memory_bytes"] == 268435456


def test_video_cpu_memory_bytes_must_be_positive(monkeypatch):
    monkeypatch.setenv("VIDEO_CPU_MEMORY_BYTES", "0")

    with pytest.raises(ValueError, match="VIDEO_CPU_MEMORY_BYTES must be positive"):
        _load_vane_main()


def test_video_gpu_runtime_uses_engine_ray_defaults(monkeypatch):
    monkeypatch.delenv("VANE_RAY_ACTOR_PREFETCH_DEPTH", raising=False)
    monkeypatch.delenv("VANE_VIDEO_RESIZE_THREADS", raising=False)

    vane_main = _load_vane_main()

    assert "cpus" not in vane_main._gpu_udf_kwargs()
    assert "ray_actor_thread_policy" not in vane_main._gpu_udf_kwargs()
    assert "VANE_RAY_ACTOR_PREFETCH_DEPTH" not in vane_main.os.environ
    assert vane_main.VIDEO_RESIZE_THREADS == 1


def test_video_gpu_transport_is_ray_data_aligned_with_resource_safe_vane_admission(monkeypatch):
    for name in (
        "VIDEO_TARGET_MAX_BLOCK_BYTES",
        "VIDEO_GPU_INPUT_HARD_MAX_BYTES",
        "VIDEO_GPU_OUTPUT_HARD_MAX_BYTES",
        "VIDEO_SOURCE_TARGET_BLOCK_BYTES",
        "VANE_VIDEO_MAX_PARTITION_BYTES",
    ):
        monkeypatch.delenv(name, raising=False)

    vane_main = _load_vane_main()
    kwargs = vane_main._gpu_udf_kwargs()

    assert vane_main.BATCH_SIZE == 32
    assert kwargs["min_task_batch_size"] == 32
    assert kwargs["target_max_batch_bytes"] == 128 * 1024**2
    assert kwargs["task_input_max_bytes"] == 192 * 1024**2
    assert kwargs["output_target_max_bytes"] == 192 * 1024**2
    assert vane_main.VIDEO_SOURCE_TARGET_BLOCK_BYTES == 128 * 1024**2


def test_video_source_target_block_bytes_has_scoped_and_legacy_overrides(monkeypatch):
    monkeypatch.delenv("VIDEO_SOURCE_TARGET_BLOCK_BYTES", raising=False)
    monkeypatch.setenv("VANE_VIDEO_MAX_PARTITION_BYTES", "67108864")
    vane_main = _load_vane_main()
    assert vane_main.VIDEO_SOURCE_TARGET_BLOCK_BYTES == 64 * 1024**2

    monkeypatch.setenv("VIDEO_SOURCE_TARGET_BLOCK_BYTES", "33554432")
    vane_main = _load_vane_main()
    assert vane_main.VIDEO_SOURCE_TARGET_BLOCK_BYTES == 32 * 1024**2


def test_video_source_target_block_bytes_must_be_positive(monkeypatch):
    monkeypatch.setenv("VIDEO_SOURCE_TARGET_BLOCK_BYTES", "0")

    with pytest.raises(ValueError, match="VIDEO_SOURCE_TARGET_BLOCK_BYTES must be positive"):
        _load_vane_main()
