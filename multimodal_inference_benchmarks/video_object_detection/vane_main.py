# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import time
import uuid
from pathlib import Path

import numpy as np
import pyarrow as pa
from ultralytics import YOLO
from video_kernels import (
    frames_to_torch_tensor,
    yolo_result_to_features,
)

import vane
from vane import image
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
PARQUET_ROW_GROUP_SIZE = int(os.environ.get("PARQUET_ROW_GROUP_SIZE", "122880"))
PARQUET_ROW_GROUP_SIZE_BYTES = os.environ.get("PARQUET_ROW_GROUP_SIZE_BYTES", "256MB").strip()

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


def _frame_batch(column) -> np.ndarray:
    if isinstance(column, pa.ChunkedArray):
        column = column.combine_chunks()
    batch = column.to_numpy_ndarray()
    expected = (len(column), FRAME_HEIGHT, FRAME_WIDTH, 3)
    if batch.shape != expected or batch.dtype != np.uint8 or not batch.flags.c_contiguous:
        raise ValueError(
            f"Unexpected frame batch: shape={batch.shape}, dtype={batch.dtype}, c_contiguous={batch.flags.c_contiguous}"
        )
    return batch


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


def main() -> None:
    start = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    con = vane.connect()
    try:
        con.execute("SET preserve_insertion_order=false")
        print(f"Parquet row groups: rows={PARQUET_ROW_GROUP_SIZE}, bytes={PARQUET_ROW_GROUP_SIZE_BYTES}")
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
        rel = rel.select(
            vane.col("frame_index"),
            vane.col("frame"),
            vane.FunctionExpression("unnest", vane.col("features")).alias("features"),
        )
        bbox = vane.FunctionExpression("struct_extract", vane.col("features"), vane.lit("bbox"))
        cropped = image.crop(vane.col("frame"), bbox)
        rel = rel.select(
            vane.col("frame_index"),
            vane.col("features"),
            image.encode(cropped, format="png").alias("object"),
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
