# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import time
import uuid
from pathlib import Path

import numpy as np
import pyarrow as pa
import torch
from PIL import Image
from torchvision.models import ResNet18_Weights, resnet18

import vane

DATA_ROOT = Path("/data/multimodal_inference_benchmarks/imagenet")
INPUT_PATH = Path(os.environ.get("INPUT_PATH", DATA_ROOT / "metadata_file.parquet")).expanduser()
IMAGE_ROOT = Path(os.environ.get("LOCAL_IMAGE_ROOT", DATA_ROOT / "train")).expanduser()
OUTPUT_DIR = Path(os.environ.get("OUTPUT_PATH", f"/tmp/vane_image_{uuid.uuid4().hex}")).expanduser()
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "100"))
NUM_GPU_NODES = int(os.environ.get("NUM_GPU_NODES", "1"))

IMAGE_SHAPE = (3, 224, 224)
IMAGE_TYPE = vane.tensor_type(vane.sqltypes.FLOAT, IMAGE_SHAPE)
WEIGHTS = ResNet18_Weights.DEFAULT
TRANSFORM = WEIGHTS.transforms()

if BATCH_SIZE <= 0 or NUM_GPU_NODES <= 0:
    raise ValueError("BATCH_SIZE and NUM_GPU_NODES must be positive")


def _parquet_input(path: Path) -> str:
    return str(path / "**/*.parquet") if path.is_dir() else str(path)


def _local_image_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() and path.exists():
        return path
    normalized = value.replace("\\", "/")
    marker = "imagenet/train/"
    relative = normalized.split(marker, 1)[1] if marker in normalized else normalized.lstrip("/")
    return IMAGE_ROOT / relative


def _decode_and_transform(table):
    image_urls = table.column("image_url").to_pylist()
    images = []
    for image_url in image_urls:
        path = _local_image_path(image_url)
        try:
            with Image.open(path) as image:
                images.append(TRANSFORM(image.convert("RGB")).numpy())
        except Exception as exc:
            raise RuntimeError(f"Failed to decode/transform image {path!s}") from exc

    batch = np.asarray(images, dtype=np.float32)
    return pa.table(
        {
            "image_url": [str(_local_image_path(value)) for value in image_urls],
            "norm_image": pa.FixedShapeTensorArray.from_numpy_ndarray(np.ascontiguousarray(batch)),
        }
    )


class ResNetModel:
    def __init__(self):
        self.device = torch.device("cuda")
        self.model = resnet18(weights=WEIGHTS).to(self.device)
        self.model.eval()

    def __call__(self, table):
        column = table.column("norm_image")
        if isinstance(column, pa.ChunkedArray):
            column = column.combine_chunks()
        batch = np.asarray(column.to_numpy_ndarray(), dtype=np.float32)
        with torch.inference_mode():
            classes = self.model(torch.from_numpy(batch).to(self.device)).argmax(dim=1).cpu().tolist()
        return pa.table(
            {
                "image_url": table.column("image_url"),
                "label": [WEIGHTS.meta["categories"][index] for index in classes],
            }
        )


def main() -> None:
    start = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    con = vane.connect()
    try:
        rel = con.read_parquet(_parquet_input(INPUT_PATH)).project("image_url")
        rel = rel.map_batches(
            _decode_and_transform,
            schema={"image_url": vane.sqltypes.VARCHAR, "norm_image": IMAGE_TYPE},
            batch_size=BATCH_SIZE,
        )
        rel = rel.map_batches(
            ResNetModel,
            schema={"image_url": vane.sqltypes.VARCHAR, "label": vane.sqltypes.VARCHAR},
            batch_size=BATCH_SIZE,
            actor_number=NUM_GPU_NODES,
            gpus=1.0,
        )
        rel.write_parquet(str(OUTPUT_DIR / "result.parquet"))
    finally:
        con.close()

    print(f"Runtime: {time.time() - start:.2f}s")


if __name__ == "__main__":
    main()
