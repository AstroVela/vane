# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import io
import os
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import pyarrow as pa
from PIL import Image
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from video_kernels import (
    crop_bbox_to_png,
    frames_to_torch_tensor,
    yolo_result_to_features,
)

import vane
from vane.datasource import read_datasource
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

FRAME_HEIGHT = 640
FRAME_WIDTH = 640
VIDEO_EXTENSIONS = {".avi", ".mkv", ".mov", ".mp4", ".webm"}
YOLO_MODEL = "yolo11n.pt"

FEATURE_ARROW_TYPE = pa.struct(
    [
        ("label", pa.int64()),
        ("confidence", pa.float64()),
        ("bbox", pa.list_(pa.float64())),
    ]
)
FEATURE_LIST_ARROW_TYPE = pa.list_(FEATURE_ARROW_TYPE)
FRAME_TYPE = vane.tensor_type(vane.sqltypes.UTINYINT, (FRAME_HEIGHT, FRAME_WIDTH, 3))
FEATURE_TYPE = vane.type("STRUCT(label BIGINT, confidence DOUBLE, bbox DOUBLE[])")
FEATURE_LIST_TYPE = vane.type("STRUCT(label BIGINT, confidence DOUBLE, bbox DOUBLE[])[]")

if BATCH_SIZE <= 0 or NUM_GPU_NODES <= 0:
    raise ValueError("BATCH_SIZE and NUM_GPU_NODES must be positive")


def _video_files(path: Path) -> list[str]:
    if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
        return [str(path)]
    files = sorted(str(file) for file in path.rglob("*") if file.suffix.lower() in VIDEO_EXTENSIONS)
    if not files:
        raise RuntimeError(f"No local video files found under {path}")
    return files


def _frame_batch(column) -> np.ndarray:
    if isinstance(column, pa.ChunkedArray):
        column = column.combine_chunks()
    batch = column.to_numpy_ndarray()
    expected = (len(column), FRAME_HEIGHT, FRAME_WIDTH, 3)
    if batch.shape != expected or batch.dtype != np.uint8:
        raise ValueError(f"Unexpected frame batch: shape={batch.shape}, dtype={batch.dtype}")
    return batch


def _feature_field(feature, name: str):
    for key, value in feature.items():
        if str(key).strip('"') == name:
            return value
    raise KeyError(name)


class YOLODetector:
    def __init__(self):
        self.model = YOLO(YOLO_MODEL)
        self.model.to("cuda")

    def __call__(self, table):
        frame_indices = table.column("frame_index").to_pylist()
        frame_column = table.column("frame")
        frames = _frame_batch(frame_column)
        tensor = frames_to_torch_tensor(frames, None)
        results = self.model(tensor, verbose=False)
        features = [yolo_result_to_features(result) for result in results]
        return pa.table(
            {
                "frame_index": pa.array(frame_indices, type=pa.int64()),
                "frame": frame_column,
                "features": pa.array(features, type=FEATURE_LIST_ARROW_TYPE),
            }
        )


def _crop_objects(table):
    frame_indices = table.column("frame_index").to_pylist()
    features = table.column("features").to_pylist()
    frames = _frame_batch(table.column("frame"))

    output_indices = []
    output_features = []
    output_objects = []
    png_buffer = io.BytesIO()
    for index, frame_features in enumerate(features):
        if not frame_features:
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

    return pa.table(
        {
            "frame_index": pa.array(output_indices, type=pa.int64()),
            "features": pa.array(output_features, type=FEATURE_ARROW_TYPE),
            "object": pa.array(output_objects, type=pa.binary()),
        }
    )


def main() -> None:
    start = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    con = vane.connect()
    try:
        rel = read_datasource(
            VideoFrameSource(
                _video_files(INPUT_PATH),
                height=FRAME_HEIGHT,
                width=FRAME_WIDTH,
            ),
            con=con,
        )
        rel = rel.map_batches(
            YOLODetector,
            schema={
                "frame_index": vane.sqltypes.BIGINT,
                "frame": FRAME_TYPE,
                "features": FEATURE_LIST_TYPE,
            },
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
        )
        rel.write_parquet(str(OUTPUT_DIR), per_thread_output=True)
    finally:
        con.close()

    print(f"Runtime: {time.time() - start:.2f}s")


if __name__ == "__main__":
    main()
