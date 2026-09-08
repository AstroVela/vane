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
import numpy.typing as npt

from vane._expressions import as_expression
from vane._image import ImageFormat, ImageMode, _image_expression

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


def resize(image: Any, w: int | vane.Expression | None, h: int | vane.Expression | None) -> vane.Expression:
    """Resize UInt8 Images with bilinear sampling and premultiplied alpha.

    Width and height must be positive integers; NULL inputs produce NULL.
    A known mode and constant target dimensions give a FixedShapeImage result.
    Per-row dimensions retain a dynamic Image with the input mode constraint.
    """
    import vane

    dimensions = []
    for name, value in (("w", w), ("h", h)):
        if value is not None and not isinstance(value, vane.Expression):
            if isinstance(value, (bool, np.bool_)):
                raise TypeError(f"resize {name} must be an integer, not bool")
            try:
                value = operator.index(value)
            except TypeError as error:
                raise TypeError(f"resize {name} must be an integer or Expression") from error
            if not 0 < value <= (1 << 32) - 1:
                raise ValueError(f"resize {name} must be a positive UINTEGER value")
        dimensions.append(as_expression(value))
    return vane.FunctionExpression("resize", _image_expression(image), *dimensions)


def convert_image(image: Any, mode: ImageMode | str | vane.Expression | None) -> vane.Expression:
    """Convert UInt8 Image colors among L, LA, RGB and RGBA without resizing.

    Alpha is preserved, added as 255, or dropped without compositing. Known
    target modes preserve fixed input dimensions; per-row modes return Image.
    """
    import vane

    if isinstance(mode, str):
        mode = str(ImageMode(mode.upper()))
    elif mode is not None and not isinstance(mode, vane.Expression):
        raise TypeError("convert_image mode must be a string, ImageMode, or Expression")
    return vane.FunctionExpression("convert_image", _image_expression(image), as_expression(mode))


def _copy_transform(pixels: memoryview, output: memoryview, check_interrupted: Callable[[], None]) -> None:
    source = np.frombuffer(pixels, dtype=np.uint8)
    target = np.frombuffer(output, dtype=np.uint8)
    block = 1024 * 1024
    for offset in range(0, source.size, block):
        check_interrupted()
        target[offset : offset + block] = source[offset : offset + block]


def _resize_image(
    pixels: memoryview,
    width: int,
    height: int,
    channels: int,
    target_width: int,
    target_height: int,
    output: memoryview,
    check_interrupted: Callable[[], None],
) -> None:
    if (width, height) == (target_width, target_height):
        _copy_transform(pixels, output, check_interrupted)
        return
    source = np.frombuffer(pixels, dtype=np.uint8).reshape(height, width, channels)
    target = np.frombuffer(output, dtype=np.uint8).reshape(-1, channels)
    alpha = channels in (2, 4)
    # Bound coordinate arrays and floating-point scratch independently of image
    # dimensions. Narrow rows do not incur a Python callback for every row.
    block = 16384
    for begin in range(0, target.shape[0], block):
        check_interrupted()
        end = min(begin + block, target.shape[0])
        indices: npt.NDArray[np.int64] = np.arange(begin, end, dtype=np.int64)
        x = np.clip((indices % target_width + 0.5) * width / target_width - 0.5, 0, width - 1)
        y = np.clip((indices // target_width + 0.5) * height / target_height - 0.5, 0, height - 1)
        x0, y0 = x.astype(np.intp), y.astype(np.intp)
        x1, y1 = np.minimum(x0 + 1, width - 1), np.minimum(y0 + 1, height - 1)
        fx, fy = x - x0, y - y0
        mixed = np.zeros((end - begin, channels), dtype=np.float64)
        for rows, columns, weight in (
            (y0, x0, (1 - fx) * (1 - fy)),
            (y0, x1, fx * (1 - fy)),
            (y1, x0, (1 - fx) * fy),
            (y1, x1, fx * fy),
        ):
            sample = source[rows, columns].astype(np.float64)
            if alpha:
                sample[:, :-1] *= sample[:, -1:]
            mixed += sample * weight[:, None]
        if alpha:
            opacity = mixed[:, -1:]
            np.divide(mixed[:, :-1], opacity, out=mixed[:, :-1], where=opacity > 0)
        np.clip(mixed, 0, 255, out=mixed)
        mixed += 0.5
        np.floor(mixed, out=mixed)
        target[begin:end] = mixed


def _convert_image(
    pixels: memoryview,
    width: int,
    height: int,
    channels: int,
    target_channels: int,
    output: memoryview,
    check_interrupted: Callable[[], None],
) -> None:
    if channels == target_channels:
        _copy_transform(pixels, output, check_interrupted)
        return
    source = np.frombuffer(pixels, dtype=np.uint8).reshape(-1, channels)
    target = np.frombuffer(output, dtype=np.uint8).reshape(-1, target_channels)
    block = 16384
    for begin in range(0, width * height, block):
        check_interrupted()
        sample = source[begin : begin + block]
        result = target[begin : begin + block]
        if target_channels < 3:
            if channels < 3:
                result[:, 0] = sample[:, 0]
            else:
                rgb = sample[:, :3].astype(np.uint32)
                result[:, 0] = (rgb[:, 0] * 299 + rgb[:, 1] * 587 + rgb[:, 2] * 114 + 500) // 1000
        elif channels < 3:
            result[:, :3] = sample[:, :1]
        else:
            result[:, :3] = sample[:, :3]
        if target_channels in (2, 4):
            result[:, -1] = sample[:, -1] if channels in (2, 4) else 255


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
    columns = min(right - left, max(1, block // channels))
    rows = max(1, block // (columns * channels))
    for row in range(top, bottom, rows):
        row_end = min(row + rows, bottom)
        for column in range(left, right, columns):
            check_interrupted()
            end = min(column + columns, right)
            target[row - y : row_end - y, column - x : end - x] = source[row:row_end, column:end]


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
