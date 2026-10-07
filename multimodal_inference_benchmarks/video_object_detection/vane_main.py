# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import io
import os
import time
import uuid
from pathlib import Path

import numpy as np
from PIL import Image
from ultralytics import YOLO
from video_kernels import (
    crop_bbox_to_png,
    frames_to_torch_tensor,
    yolo_result_to_features,
)

import vane
from vane.datasource import DataSource, read_datasource
from vane.datasource.video_reader import VideoFrameSource

INPUT_PATH = Path(
    os.environ.get(
        "INPUT_PATH",
        "/data/multimodal_inference_benchmarks/hollywood2/AVIClips",
    )
).expanduser()
OUTPUT_DIR = Path(os.environ.get("OUTPUT_PATH", f"/tmp/vane_video_{uuid.uuid4().hex}")).expanduser()
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "32"))
NUM_GPU_NODES = int(os.environ.get("NUM_GPU_NODES", "1"))
PARQUET_ROW_GROUP_SIZE = int(os.environ.get("PARQUET_ROW_GROUP_SIZE", "122880"))
PARQUET_ROW_GROUP_SIZE_BYTES = os.environ.get("PARQUET_ROW_GROUP_SIZE_BYTES", "256MB").strip()

FRAME_HEIGHT = 640
FRAME_WIDTH = 640
VIDEO_EXTENSIONS = {".avi", ".mkv", ".mov", ".mp4", ".webm"}
YOLO_MODEL = "yolo11n.pt"

FRAME_TYPE = vane.tensor_type(vane.sqltypes.UTINYINT, (FRAME_HEIGHT, FRAME_WIDTH, 3))
FEATURE_TYPE = vane.type("STRUCT(label BIGINT, confidence DOUBLE, bbox DOUBLE[])")
FEATURE_LIST_TYPE = vane.type("STRUCT(label BIGINT, confidence DOUBLE, bbox DOUBLE[])[]")

if min(BATCH_SIZE, NUM_GPU_NODES, PARQUET_ROW_GROUP_SIZE) <= 0:
    raise ValueError("BATCH_SIZE, NUM_GPU_NODES, and PARQUET_ROW_GROUP_SIZE must be positive")
if not PARQUET_ROW_GROUP_SIZE_BYTES:
    raise ValueError("PARQUET_ROW_GROUP_SIZE_BYTES must be non-empty")


def _video_files(path: Path) -> list[str]:
    if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
        return [str(path)]
    files = sorted(str(file) for file in path.rglob("*") if file.suffix.lower() in VIDEO_EXTENSIONS)
    if not files:
        raise RuntimeError(f"No local video files found under {path}")
    return files


class PythonVideoFrameSource(DataSource):
    """Keep Python decoding's uint8 tensor schema at the benchmark boundary."""

    def __init__(self, files, **options):
        self.source = VideoFrameSource(files, **options)

    @property
    def schema(self):
        return self.source.schema

    def get_tasks(self):
        return self.source.get_tasks()


def _frame_batch(frames: np.ndarray) -> np.ndarray:
    """Check the model's RGB/size contract after framework batch conversion."""
    if frames.ndim == 1 and frames.dtype == object:
        for frame in frames:
            if frame is None:
                raise ValueError("Video frames cannot contain NULL values")
            if frame.shape != (FRAME_HEIGHT, FRAME_WIDTH, 3) or frame.dtype != np.uint8:
                raise ValueError(f"Unexpected frame: shape={frame.shape}, dtype={frame.dtype}")
        return frames
    expected = (len(frames), FRAME_HEIGHT, FRAME_WIDTH, 3)
    if frames.shape != expected or frames.dtype != np.uint8:
        raise ValueError(f"Unexpected frame batch: shape={frames.shape}, dtype={frames.dtype}")
    return frames


def _feature_field(feature, name: str):
    for key, value in feature.items():
        if str(key).strip('"') == name:
            return value
    raise KeyError(name)


class YOLODetector:
    def __init__(self):
        self.model = YOLO(YOLO_MODEL)
        self.model.to("cuda")

    def __call__(self, batch):
        frames = _frame_batch(batch["frame"])
        features = np.empty(len(frames), dtype=object)
        if len(frames):
            tensor = frames_to_torch_tensor(frames, None)
            results = self.model(tensor, verbose=False)
            for index, result in zip(range(len(frames)), results, strict=True):
                features[index] = yolo_result_to_features(result)
        return {"frame_index": batch["frame_index"], "frame": frames, "features": features}


def _crop_objects(batch):
    frame_indices = batch["frame_index"]
    features = batch["features"]
    frames = _frame_batch(batch["frame"])

    output_indices = []
    output_features = []
    output_objects = []
    png_buffer = io.BytesIO()
    for index, frame_features in enumerate(features):
        if frame_features is None or frame_features is np.ma.masked or len(frame_features) == 0:
            continue
        image = Image.fromarray(frames[index])
        for feature in frame_features:
            output_indices.append(frame_indices[index])
            output_features.append(feature)
            output_objects.append(
                crop_bbox_to_png(
                    frames[index],
                    _feature_field(feature, "bbox"),
                    pil_image=image,
                    png_buffer=png_buffer,
                )
            )

    return {
        "frame_index": np.asarray(output_indices, dtype=np.int64),
        "features": np.asarray(output_features, dtype=object),
        "object": np.asarray(output_objects, dtype=object),
    }


def main() -> None:
    start = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    con = vane.connect()
    try:
        con.execute("SET video_backend='python'")
        con.execute("SET image_backend='python'")
        con.execute("SET preserve_insertion_order=false")
        print(f"Parquet row groups: rows={PARQUET_ROW_GROUP_SIZE}, bytes={PARQUET_ROW_GROUP_SIZE_BYTES}")
        rel = read_datasource(
            PythonVideoFrameSource(
                _video_files(INPUT_PATH),
                height=FRAME_HEIGHT,
                width=FRAME_WIDTH,
            ),
            con=con,
        ).project("frame_index, frame")
        rel = rel.map_batches(
            YOLODetector,
            schema={
                "frame_index": vane.sqltypes.BIGINT,
                "frame": FRAME_TYPE,
                "features": FEATURE_LIST_TYPE,
            },
            batch_format="numpy",
            batch_size=BATCH_SIZE,
            actor_number=NUM_GPU_NODES,
            gpus=1.0,
        )
        rel = rel.map_batches(
            _crop_objects,
            schema={
                "frame_index": vane.sqltypes.BIGINT,
                "features": FEATURE_TYPE,
                "object": vane.sqltypes.BLOB,
            },
            batch_format="numpy",
        )
        rel.write_parquet(
            str(OUTPUT_DIR),
            per_thread_output=True,
            row_group_size=PARQUET_ROW_GROUP_SIZE,
            row_group_size_bytes=PARQUET_ROW_GROUP_SIZE_BYTES,
        )
    finally:
        con.close()

    print(f"Runtime: {time.time() - start:.2f}s")


if __name__ == "__main__":
    main()
