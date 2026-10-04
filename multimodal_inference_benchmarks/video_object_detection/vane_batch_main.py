# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Experimental BatchUDF detector; the original video entrypoint is unchanged."""

from vane_main import YOLODetector as OriginalYOLODetector
from vane_main import _frame_batch, main
from video_kernels import frames_to_torch_tensor, yolo_result_to_features

from vane.udf import BatchUDF


class YOLODetector(OriginalYOLODetector, BatchUDF):
    """Keep input adaptation and output encoding outside model computation."""

    def prepare_batch(self, table):
        frame_indices = table.column("frame_index").to_pylist()
        frame_column = table.column("frame")
        # Reuse a contiguous single chunk, as Ray's tensor batch formatter does.
        if frame_column.num_chunks == 1:
            frame_column = frame_column.chunk(0)
        frames = _frame_batch(frame_column)
        return {"frame_index": frame_indices, "frame": frames}

    def __call__(self, batch):
        frames = batch["frame"]
        tensor = frames_to_torch_tensor(frames, None)
        results = self.model(tensor, verbose=False)
        features = [yolo_result_to_features(result) for result in results]
        return {"frame_index": batch["frame_index"], "frame": frames, "features": features}


if __name__ == "__main__":
    main(detector_cls=YOLODetector)
