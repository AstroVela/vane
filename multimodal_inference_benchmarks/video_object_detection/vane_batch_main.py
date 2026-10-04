# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Ordinary callable-class detector with framework-encoded NumPy outputs."""

from vane_main import YOLODetector as OriginalYOLODetector
from vane_main import _frame_batch, main
from video_kernels import frames_to_torch_tensor, yolo_result_to_features


class YOLODetector(OriginalYOLODetector):
    """Return materialized columns; the runtime owns threads and Arrow encoding."""

    def __call__(self, table):
        frame_indices = table.column("frame_index").to_pylist()
        frames = _frame_batch(table.column("frame"))
        tensor = frames_to_torch_tensor(frames, None)
        results = self.model(tensor, verbose=False)
        features = [yolo_result_to_features(result) for result in results]
        return {"frame_index": frame_indices, "frame": frames, "features": features}


if __name__ == "__main__":
    main(detector_cls=YOLODetector)
