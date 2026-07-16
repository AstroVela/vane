from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_VIDEO_DIR = _REPO_ROOT / "multimodal_inference_benchmarks/video_object_detection"


def _video_kernels():
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    pytest.importorskip("PIL")
    if str(_VIDEO_DIR) not in sys.path:
        sys.path.insert(0, str(_VIDEO_DIR))
    import video_kernels

    return video_kernels


def _video_inputs():
    if str(_VIDEO_DIR) not in sys.path:
        sys.path.insert(0, str(_VIDEO_DIR))
    import video_inputs

    return video_inputs


def test_video_reader_decode_errors_are_not_silently_swallowed(monkeypatch):
    from duckdb.datasource.video_reader import VideoFrameTask

    def fail_decode(_self):
        raise RuntimeError("decode failed")

    monkeypatch.setattr(VideoFrameTask, "_execute_inner", fail_decode)

    with pytest.raises(RuntimeError, match="decode failed"):
        list(VideoFrameTask("bad.avi", height=8, width=8).execute())


def test_video_gpu_transport_defaults_match_ray_data_task_shape(monkeypatch):
    kernels = _video_kernels()

    for name in (
        "VIDEO_TARGET_MAX_BLOCK_BYTES",
        "VIDEO_GPU_INPUT_HARD_MAX_BYTES",
        "VIDEO_GPU_OUTPUT_HARD_MAX_BYTES",
    ):
        monkeypatch.delenv(name, raising=False)

    config = kernels.video_gpu_transport_config_from_env()

    assert config == kernels.VideoGpuTransportConfig(
        target_max_block_bytes=128 * 1024**2,
        input_hard_max_bytes=192 * 1024**2,
        output_hard_max_bytes=192 * 1024**2,
    )


def test_video_read_task_count_matches_ray_data_default_parallelism():
    inputs = _video_inputs()
    paths = [f"clip-{index}.avi" for index in range(1000)]

    assert inputs.ray_data_read_task_count(
        paths,
        input_size_bytes=int(5.3 * 1024**3),
        available_cpus=36,
        target_max_block_size=128 * 1024**2,
    ) == 200
    assert inputs.ray_data_read_task_count(
        paths,
        input_size_bytes=50 * 1024**2,
        available_cpus=36,
        target_max_block_size=128 * 1024**2,
    ) == 72
    assert inputs.ray_data_read_task_count(
        paths,
        input_size_bytes=int(5.3 * 1024**3),
        available_cpus=36,
        target_max_block_size=128 * 1024**2,
        input_limit=256,
    ) == 1


def test_video_gpu_transport_hard_limit_covers_ray_data_soft_target(monkeypatch):
    kernels = _video_kernels()
    monkeypatch.setenv("VIDEO_TARGET_MAX_BLOCK_BYTES", "1024")
    monkeypatch.setenv("VIDEO_GPU_OUTPUT_HARD_MAX_BYTES", "1000")

    with pytest.raises(ValueError, match="must be at least"):
        kernels.video_gpu_transport_config_from_env()

    monkeypatch.setenv("VIDEO_GPU_OUTPUT_HARD_MAX_BYTES", "1536")
    monkeypatch.setenv("VIDEO_GPU_INPUT_HARD_MAX_BYTES", "1000")
    with pytest.raises(ValueError, match="VIDEO_GPU_INPUT_HARD_MAX_BYTES"):
        kernels.video_gpu_transport_config_from_env()


def test_shared_video_tensor_kernel_matches_reference():
    torch = pytest.importorskip("torch")
    import torchvision
    from PIL import Image

    kernels = _video_kernels()
    frames = np.arange(2 * 4 * 5 * 3, dtype=np.uint8).reshape(2, 4, 5, 3)

    actual = kernels.frames_to_torch_tensor(frames, torch.device("cpu"))
    expected = torch.stack(
        [torchvision.transforms.functional.to_tensor(Image.fromarray(frame)) for frame in frames],
        dim=0,
    )

    torch.testing.assert_close(actual, expected)


def test_shared_video_tensor_kernel_preserves_original_ray_data_cpu_boundary():
    torch = pytest.importorskip("torch")
    import torchvision
    from PIL import Image

    kernels = _video_kernels()
    frames = np.arange(2 * 4 * 5 * 3, dtype=np.uint8).reshape(2, 4, 5, 3)

    actual = kernels.frames_to_torch_tensor(frames, None)
    expected = torch.stack(
        [torchvision.transforms.functional.to_tensor(Image.fromarray(frame)) for frame in frames],
        dim=0,
    )

    assert actual.device.type == "cpu"
    torch.testing.assert_close(actual, expected)


def test_shared_resize_matches_vane_video_reader():
    from duckdb.datasource.video_reader import _resize_rgb_frame

    kernels = _video_kernels()
    frame = np.arange(3 * 4 * 3, dtype=np.uint8).reshape(3, 4, 3)

    expected = _resize_rgb_frame(frame, width=7, height=5)
    actual = kernels.resize_rgb_frame(frame, width=7, height=5)

    np.testing.assert_array_equal(actual, expected)


def test_shared_crop_png_matches_reference_bytes():
    from PIL import Image

    kernels = _video_kernels()
    frame = np.arange(8 * 9 * 3, dtype=np.uint8).reshape(8, 9, 3)
    bbox = [1.0, 2.0, 7.0, 6.0]
    reference_buffer = io.BytesIO()
    Image.fromarray(frame).crop((1, 2, 7, 6)).save(reference_buffer, format="PNG", compress_level=2)

    assert kernels.crop_bbox_to_png(frame, bbox) == reference_buffer.getvalue()


def test_benchmark_entrypoints_use_shared_kernels_and_engine_actor_defaults():
    vane_source = (_VIDEO_DIR / "vane_main.py").read_text(encoding="utf-8")
    ray_data_source = (_VIDEO_DIR / "ray_data_main.py").read_text(encoding="utf-8")

    for source in (vane_source, ray_data_source):
        assert "frames_to_torch_tensor" in source
        assert "yolo_result_to_features" in source
        assert "crop_bbox_to_png" in source
    assert "configure_video_torch_threads" not in vane_source
    assert "ray_actor_uses_native_threads" not in vane_source
    assert '"ray_actor_thread_policy"' not in vane_source
    assert '"cpus": VIDEO_GPU_CPUS' not in vane_source
    assert "VANE_RAY_ACTOR_PREFETCH_DEPTH" not in vane_source
    assert "num_cpus" not in ray_data_source
    assert "concurrency=NUM_GPU_NODES" in ray_data_source


def test_benchmark_entrypoints_preserve_original_ray_data_yolo_call_boundary():
    vane_source = (_VIDEO_DIR / "vane_main.py").read_text(encoding="utf-8")
    ray_data_source = (_VIDEO_DIR / "ray_data_main.py").read_text(encoding="utf-8")

    for source in (vane_source, ray_data_source):
        assert "self.model(stack, verbose=True)" in source
        assert "self.model(stack, verbose=False)" not in source
        assert "frames_to_torch_tensor(frames, None" in source


def test_video_benchmarks_preserve_full_frame_output_and_align_actor_task_transport():
    vane_source = (_VIDEO_DIR / "vane_main.py").read_text(encoding="utf-8")
    ray_data_source = (_VIDEO_DIR / "ray_data_main.py").read_text(encoding="utf-8")

    assert '"frame": FRAME_SQL_TYPE' in vane_source
    assert 'batch["features"] = features' in ray_data_source
    assert "return batch" in ray_data_source
    # Ray Data preserves a large physical block, but coalesces undersized
    # blocks until the input bundle reaches one compute batch.
    assert '"min_task_batch_size": BATCH_SIZE' in vane_source
    assert '"task_input_max_bytes": VIDEO_GPU_TRANSPORT.input_hard_max_bytes' in vane_source
    assert '"output_target_max_bytes": VIDEO_GPU_TRANSPORT.output_hard_max_bytes' in vane_source
    assert "max_partition_bytes=VIDEO_SOURCE_TARGET_BLOCK_BYTES" in vane_source
    assert "max_tasks_in_flight_per_actor" not in ray_data_source
