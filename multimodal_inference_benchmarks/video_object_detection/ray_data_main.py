# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import time
import uuid
from pathlib import Path

import ray
import torch
from batch_profile import BatchProfile
from ultralytics import YOLO
from video_kernels import (
    crop_bbox_to_png,
    frames_to_torch_tensor,
    resize_rgb_frame,
    yolo_result_to_features,
)

INPUT_PATH = Path(
    os.environ.get(
        "INPUT_PATH",
        "/data/multimodal_inference_benchmarks/hollywood2/AVIClips",
    )
).expanduser()
OUTPUT_PATH = Path(os.environ.get("OUTPUT_PATH", f"/tmp/ray_data_video_{uuid.uuid4().hex}")).expanduser()
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "32"))
NUM_GPU_NODES = int(os.environ.get("NUM_GPU_NODES", "1"))
INPUT_LIMIT = int(os.environ.get("INPUT_LIMIT", "0"))
IMAGE_HEIGHT = int(os.environ.get("IMAGE_HEIGHT", "640"))
IMAGE_WIDTH = int(os.environ.get("IMAGE_WIDTH", "640"))
YOLO_MODEL = os.environ.get("YOLO_MODEL", "yolo11n.pt")

if BATCH_SIZE <= 0 or NUM_GPU_NODES <= 0:
    raise ValueError("BATCH_SIZE and NUM_GPU_NODES must be positive")


class ExtractImageFeatures:
    def __init__(self):
        self.model = YOLO(YOLO_MODEL)
        if torch.cuda.is_available():
            self.model.to("cuda")
        self.profile = BatchProfile.from_env("ray_data")

    def __call__(self, batch):
        profile = self.profile
        if profile:
            profile.begin(len(batch["frame"]))
        frames = batch["frame"]
        if len(frames) == 0:
            batch["features"] = []
            if profile:
                profile.mark("array")
                profile.finish([])
            return batch
        if profile:
            profile.mark("array")
        tensor = frames_to_torch_tensor(frames, None)
        if profile:
            profile.mark("cpu_tensor")
        results = self.model(tensor, verbose=False)
        if profile:
            profile.mark("model")
        features = [yolo_result_to_features(result) for result in results]
        if profile:
            profile.mark("features")
        batch["features"] = features
        if profile:
            profile.mark("pack")
            profile.finish(results)
        return batch


def resize_frame(row):
    row["frame"] = resize_rgb_frame(
        row["frame"],
        width=IMAGE_WIDTH,
        height=IMAGE_HEIGHT,
    )
    return row


def explode_features(row):
    for feature in row["features"]:
        row["features"] = feature
        yield row


def crop_image(row):
    row["object"] = crop_bbox_to_png(row["frame"], row["features"]["bbox"])
    return row


def main() -> None:
    ray.init(ignore_reinit_error=True)

    start = time.time()
    ds = ray.data.read_videos(str(INPUT_PATH))
    if INPUT_LIMIT > 0:
        ds = ds.limit(INPUT_LIMIT)
    ds = ds.map(resize_frame)
    ds = ds.map_batches(
        ExtractImageFeatures,
        batch_size=BATCH_SIZE,
        num_gpus=1.0,
        concurrency=NUM_GPU_NODES,
    )
    ds = ds.flat_map(explode_features)
    ds = ds.map(crop_image)
    ds = ds.drop_columns(["frame"])
    ds.write_parquet(str(OUTPUT_PATH))

    print(f"Runtime: {time.time() - start:.2f}s")


if __name__ == "__main__":
    main()
