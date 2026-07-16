import warnings
import inspect

import numpy as np
import pytest
import torch
import torchvision
from PIL import Image


def test_frames_to_tensor_batch_matches_torchvision_reference_for_numpy_batch():
    from multimodal_inference_benchmarks.video_object_detection.video_tensor import frames_to_tensor_batch

    frames = np.arange(2 * 4 * 5 * 3, dtype=np.uint8).reshape(2, 4, 5, 3)

    actual = frames_to_tensor_batch(frames)
    expected = torch.stack(
        [torchvision.transforms.functional.to_tensor(Image.fromarray(frame)) for frame in frames],
        dim=0,
    )

    assert actual.dtype == torch.float32
    assert actual.shape == (2, 3, 4, 5)
    torch.testing.assert_close(actual, expected)


def test_frames_to_tensor_batch_matches_torchvision_reference_for_frame_list():
    from multimodal_inference_benchmarks.video_object_detection.video_tensor import frames_to_tensor_batch

    frames = [
        np.full((3, 2, 3), 17, dtype=np.uint8),
        np.full((3, 2, 3), 251, dtype=np.uint8),
    ]

    actual = frames_to_tensor_batch(frames)
    expected = torch.stack(
        [torchvision.transforms.functional.to_tensor(Image.fromarray(frame)) for frame in frames],
        dim=0,
    )

    torch.testing.assert_close(actual, expected)


def test_frames_to_tensor_batch_respects_requested_device():
    from multimodal_inference_benchmarks.video_object_detection.video_tensor import frames_to_tensor_batch

    frames = np.zeros((2, 4, 5, 3), dtype=np.uint8)

    result = frames_to_tensor_batch(frames, device=torch.device("cpu"))

    assert result.device.type == "cpu"
def test_frames_to_tensor_batch_rejects_non_rgb_uint8_frames():
    from multimodal_inference_benchmarks.video_object_detection.video_tensor import frames_to_tensor_batch

    with pytest.raises(ValueError, match="uint8 RGB"):
        frames_to_tensor_batch(np.zeros((2, 4, 5, 1), dtype=np.uint8))

    with pytest.raises(ValueError, match="uint8 RGB"):
        frames_to_tensor_batch(np.zeros((2, 4, 5, 3), dtype=np.float32))


def test_frames_to_tensor_batch_copies_readonly_batches_without_warning():
    from multimodal_inference_benchmarks.video_object_detection.video_tensor import frames_to_tensor_batch

    frames = np.zeros((2, 4, 5, 3), dtype=np.uint8)
    frames.setflags(write=False)

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        result = frames_to_tensor_batch(frames)

    assert result.shape == (2, 3, 4, 5)
