# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import time
import uuid
from pathlib import Path

import pyarrow as pa
from ultralytics import YOLO
from vane_image_pipeline import FRAME_HEIGHT, FRAME_TYPE, FRAME_WIDTH, crop_objects, frame_batch
from video_kernels import (
    frames_to_torch_tensor,
    yolo_result_to_features,
)

import vane

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


class YOLODetector:
    def __init__(self):
        self.model = YOLO(YOLO_MODEL)
        self.model.to("cuda")

    def __call__(self, table):
        frame_indices = table.column("frame_index").to_pylist()
        frame_column = table.column("frame")
        frames = frame_batch(frame_column)
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
        vane.load_installed_extension("native_media", connection=con)
        con.execute("SET image_backend='native'")
        con.execute("SET preserve_insertion_order=false")
        print(f"Parquet row groups: rows={PARQUET_ROW_GROUP_SIZE}, bytes={PARQUET_ROW_GROUP_SIZE_BYTES}")
        rel = vane.read_video_frames(
            _video_files(INPUT_PATH),
            image_height=FRAME_HEIGHT,
            image_width=FRAME_WIDTH,
            connection=con,
        ).project("frame_index, data AS frame")
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
        rel = crop_objects(rel)
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
