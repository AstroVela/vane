# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Image expressions and the independent Python pixel/codec implementation."""

from __future__ import annotations

import importlib
import io
import operator
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np

from vane._expressions import as_expression
from vane._image import ImageFormat, _image_expression

if TYPE_CHECKING:
    import vane


def crop(image: Any, bbox: tuple[int, int, int, int] | list[int] | vane.Expression) -> vane.Expression:
    """Crop an Image using integer (x, y, width, height) coordinates.

    Width and height must be positive. Pixels outside the input are zero-filled,
    including alpha channels. The result has dynamic dimensions and retains the
    input's mode constraint. Neither pixels nor coordinates are implicitly cast.
    """
    import vane

    if isinstance(bbox, (tuple, list)):
        if len(bbox) != 4:
            raise ValueError("crop bbox must contain exactly four integers: x, y, width, height")
        if any(isinstance(value, (bool, np.bool_)) for value in bbox):
            raise TypeError("crop bbox coordinates must be integers, not bool")
        try:
            bbox = [operator.index(value) for value in bbox]
        except TypeError as error:
            raise TypeError("crop bbox coordinates must be integers") from error
    return vane.FunctionExpression("crop", _image_expression(image), as_expression(bbox))


def encode_image(image: Any, image_format: ImageFormat | str | vane.Expression) -> vane.Expression:
    """Encode an Image as PNG bytes, preserving its UInt8 mode and pixels."""
    import vane

    if isinstance(image_format, ImageFormat):
        image_format = str(image_format)
    return vane.FunctionExpression("encode_image", _image_expression(image), as_expression(image_format))


def _crop_image(
    pixels: memoryview,
    width: int,
    height: int,
    channels: int,
    x: int,
    y: int,
    crop_width: int,
    crop_height: int,
    output: memoryview,
    check_interrupted: Callable[[], None],
) -> None:
    """Operate on borrowed engine buffers only for the duration of this call."""
    source = np.frombuffer(pixels, dtype=np.uint8).reshape(height, width, channels)
    target = np.frombuffer(output, dtype=np.uint8).reshape(crop_height, crop_width, channels)
    flat = target.reshape(-1)
    block = 1024 * 1024
    for offset in range(0, flat.size, block):
        check_interrupted()
        flat[offset : offset + block] = 0
    left, top = max(x, 0), max(y, 0)
    right, bottom = min(x + crop_width, width), min(y + crop_height, height)
    if right <= left or bottom <= top:
        return
    columns = max(1, block // channels)
    for row in range(top, bottom):
        for column in range(left, right, columns):
            check_interrupted()
            end = min(column + columns, right)
            target[row - y, column - x : end - x] = source[row, column:end]


class _PNGBuffer(io.BytesIO):
    def __init__(self, limit: int, check_interrupted: Callable[[], None]) -> None:
        super().__init__()
        self.limit = limit
        self.check_interrupted = check_interrupted

    def write(self, data: Any) -> int:
        self.check_interrupted()
        if len(data) > self.limit - self.tell():
            raise OverflowError("PNG encoding exceeds the Image operator batch byte limit")
        return super().write(data)


def _encode_image_png(
    pixels: memoryview,
    width: int,
    height: int,
    channels: int,
    max_output_bytes: int,
    check_interrupted: Callable[[], None],
) -> bytes:
    image_module = importlib.import_module("PIL.Image")

    check_interrupted()
    data = np.frombuffer(pixels, dtype=np.uint8).reshape(height, width, channels)
    if channels == 1:
        data = data[:, :, 0]
    with image_module.fromarray(data) as image, _PNGBuffer(max_output_bytes, check_interrupted) as output:
        image.save(output, format="PNG")
        check_interrupted()
        return output.getvalue()
