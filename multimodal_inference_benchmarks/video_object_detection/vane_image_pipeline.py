# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""CPU-side Image boundaries for the video detection benchmark."""

from __future__ import annotations

import numpy as np
import pyarrow as pa

import vane
from vane._image import image_arrow_type

FRAME_HEIGHT = 640
FRAME_WIDTH = 640
FRAME_TYPE = vane.image_type("RGB", FRAME_HEIGHT, FRAME_WIDTH)


def frame_batch(column: pa.Array | pa.ChunkedArray) -> np.ndarray:
    if not column.type.equals(image_arrow_type(FRAME_TYPE)):
        raise ValueError("Video detection requires IMAGE('RGB', 640, 640) frames")
    if column.null_count:
        raise ValueError("Video detection does not accept NULL frames")
    if isinstance(column, pa.ChunkedArray):
        column = column.combine_chunks()
    pixels = column.storage.flatten()
    if pixels.null_count:
        raise ValueError("Video detection does not accept NULL pixels")
    return pixels.to_numpy(zero_copy_only=True).reshape(len(column), FRAME_HEIGHT, FRAME_WIDTH, 3)


def crop_objects(relation: vane.DuckDBPyRelation) -> vane.DuckDBPyRelation:
    # YOLO emits floating-point (left, top, right, bottom). Truncate each
    # coordinate explicitly, then convert to the Image API's (x, y, w, h).
    objects = relation.project("frame_index, frame, unnest(features) AS features")
    boxes = objects.project(
        "frame_index, frame, features, list_transform(features.bbox, value -> trunc(value)::BIGINT) AS bounds"
    )
    return boxes.project(
        "frame_index, features, "
        "encode_image(crop(frame, [bounds[1], bounds[2], "
        "bounds[3] - bounds[1], bounds[4] - bounds[2]]), 'PNG') AS object"
    )
