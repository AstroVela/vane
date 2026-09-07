# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Decoded uint8 HWC Image values, metadata, and Arrow transport."""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping
from enum import Enum
from typing import Any, TypeAlias

import numpy as np
import numpy.typing as npt
import pyarrow as pa

import vane
from vane._expressions import as_expression

Image: TypeAlias = npt.NDArray[np.uint8]


class _ImageStringEnum(str, Enum):
    def __str__(self) -> str:
        return str(self.value)


class ImageMode(_ImageStringEnum):
    L = "L"
    LA = "LA"
    RGB = "RGB"
    RGBA = "RGBA"


class ImageFormat(_ImageStringEnum):
    PNG = "PNG"
    JPEG = "JPEG"
    TIFF = "TIFF"
    GIF = "GIF"
    BMP = "BMP"


class ImageProperty(_ImageStringEnum):
    Height = "height"
    Width = "width"
    Channel = "channel"
    Mode = "mode"


_MODE_CODES = {"L": 1, "LA": 2, "RGB": 3, "RGBA": 4}
_MODE_NAMES = {code: name for name, code in _MODE_CODES.items()}
_IMAGE_FIELDS = ("data", "channel", "height", "width", "mode")
_EXTENSION_NAME = "vane.image"
_DYNAMIC_STORAGE = pa.struct(
    [
        ("data", pa.list_(pa.uint8())),
        ("channel", pa.uint16()),
        ("height", pa.uint32()),
        ("width", pa.uint32()),
        ("mode", pa.uint8()),
    ]
)


def _metadata_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate Image metadata key: {key}")
        result[key] = value
    return result


class _ImageArrowType(pa.ExtensionType):  # type: ignore[misc]  # PyArrow does not ship extension type stubs.
    def __init__(self, mode: str | None = None, height: int | None = None, width: int | None = None) -> None:
        if mode is not None:
            if not isinstance(mode, str):
                raise ValueError("Image mode must be a string or None")
            mode = str(ImageMode(mode))
        if (height is None) != (width is None):
            raise ValueError("Image height and width must be provided together")
        self.mode = mode
        self.height = height
        self.width = width
        storage = _DYNAMIC_STORAGE
        if height is not None:
            if mode is None or type(height) is not int or type(width) is not int or height <= 0 or width <= 0:
                raise ValueError("Fixed Image requires a mode and positive integer dimensions")
            size = height * width * _MODE_CODES[mode]
            if size > (1 << 31) - 1:
                raise ValueError("Fixed Image cannot exceed 2147483647 pixel values")
            storage = pa.list_(pa.uint8(), size)
        super().__init__(storage, _EXTENSION_NAME)

    def __arrow_ext_serialize__(self) -> bytes:
        return json.dumps(
            {"mode": self.mode, "height": self.height, "width": self.width}, separators=(",", ":")
        ).encode()

    @classmethod
    def __arrow_ext_deserialize__(cls, storage_type: pa.DataType, serialized: bytes) -> _ImageArrowType:
        if len(serialized) > 512:
            raise ValueError("Image Arrow metadata exceeds 512 bytes")
        metadata = json.loads(serialized, object_pairs_hook=_metadata_object)
        if not isinstance(metadata, dict) or set(metadata) != {"mode", "height", "width"}:
            raise ValueError("Image Arrow metadata requires exactly mode, height, and width")
        result = cls(**metadata)
        if not storage_type.equals(result.storage_type):
            raise ValueError("Image Arrow storage does not match its mode and dimensions")
        return result

    def __reduce__(self) -> tuple[Any, tuple[Any, ...]]:
        return _ImageArrowType, (self.mode, self.height, self.width)


pa.register_extension_type(_ImageArrowType())


def image_arrow_type(dtype: Any) -> _ImageArrowType:
    mode = dtype.image_mode
    if dtype.is_fixed_shape_image():
        return _ImageArrowType(mode, *dtype.shape)
    return _ImageArrowType(mode)


def _validate_image_arrow_type(actual: pa.DataType, dtype: Any, *, boundary: str) -> _ImageArrowType:
    expected = image_arrow_type(dtype)
    if isinstance(actual, pa.BaseExtensionType):
        if actual.extension_name != _EXTENSION_NAME:
            raise vane.InvalidInputException(f"{boundary} expected vane.image Arrow type")
        try:
            decoded = _ImageArrowType.__arrow_ext_deserialize__(actual.storage_type, actual.__arrow_ext_serialize__())
        except (TypeError, ValueError) as exc:
            raise vane.InvalidInputException(f"{boundary} invalid Image Arrow metadata: {exc}") from exc
        if decoded.__arrow_ext_serialize__() != expected.__arrow_ext_serialize__():
            raise vane.InvalidInputException(f"{boundary} Image mode or dimensions do not match {dtype}")
    elif not actual.equals(expected.storage_type):
        raise vane.InvalidInputException(f"{boundary} Image requires canonical {expected.storage_type}, got {actual}")
    return expected


def _validate_layout(dtype: Any, width: int, height: int, mode: str) -> None:
    if width <= 0 or height <= 0 or width > (1 << 32) - 1 or height > (1 << 32) - 1:
        raise vane.InvalidInputException("Image width and height must be positive UInt32 values")
    if dtype.image_mode is not None and mode != dtype.image_mode:
        raise vane.InvalidInputException(f"Image mode {mode} does not match {dtype}")
    if dtype.is_fixed_shape_image() and (height, width) != dtype.shape:
        raise vane.InvalidInputException(f"Image shape {(height, width)} does not match {dtype}")


def _image_native_storage(value: Any, dtype: Any) -> Any:
    mode = None
    pil = sys.modules.get("PIL.Image")
    if pil is not None and isinstance(value, pil.Image):
        mode = value.mode
        if mode not in _MODE_CODES:
            raise vane.InvalidInputException(f"Unsupported Image mode {mode!r}")
        value = np.asarray(value)
        if mode == "L":
            value = value[:, :, np.newaxis]
    if not isinstance(value, np.ndarray) or isinstance(value, np.ma.MaskedArray):
        raise vane.InvalidInputException("Image input must be an HWC uint8 ndarray or PIL.Image.Image")
    if value.dtype != np.uint8 or value.ndim != 3 or not 1 <= value.shape[2] <= 4:
        raise vane.InvalidInputException("Image requires uint8 HWC pixels with 1 to 4 channels")
    height, width, channels = value.shape
    mode = _MODE_NAMES[channels] if mode is None else mode
    _validate_layout(dtype, width, height, mode)
    # Arrow's UInt8 list builder consumes NumPy buffers directly. Expanding a
    # 4K image into Python list entries would multiply its memory footprint.
    pixels = value.ravel(order="C")
    if dtype.is_fixed_shape_image():
        return pixels
    return {"data": pixels, "channel": channels, "height": height, "width": width, "mode": _MODE_CODES[mode]}


def _image_storage_to_numpy(value: Any, dtype: Any) -> Image:
    if dtype.is_fixed_shape_image():
        height, width = dtype.shape
        mode = str(dtype.image_mode)
        pixels = value
    else:
        if not isinstance(value, Mapping) or set(value) != set(_IMAGE_FIELDS):
            raise vane.InvalidInputException("Image storage requires the five canonical Image fields")
        if any(value[field] is None for field in _IMAGE_FIELDS):
            raise vane.InvalidInputException("Non-NULL Image cannot contain NULL fields")
        stored_mode = _MODE_NAMES.get(value["mode"])
        if stored_mode is None or value["channel"] != _MODE_CODES[stored_mode]:
            raise vane.InvalidInputException("Image mode and channel count do not match")
        mode = stored_mode
        height, width, pixels = value["height"], value["width"], value["data"]
    _validate_layout(dtype, width, height, mode)
    channels = _MODE_CODES[mode]
    if len(pixels) != height * width * channels or any(
        type(pixel) is not int or not 0 <= pixel <= 255 for pixel in pixels
    ):
        raise vane.InvalidInputException("Image pixels must be non-NULL UInt8 values matching its HWC shape")
    return np.array(pixels, dtype=np.uint8).reshape(height, width, channels)


def _image_expression(value: Any) -> vane.Expression:
    if isinstance(value, np.ndarray):
        return vane.ConstantExpression(vane.Value(value, vane.image_type()))
    return as_expression(value)


def image_attribute(image: Any, name: ImageProperty | str | vane.Expression) -> vane.Expression:
    if isinstance(name, str):
        name = str(ImageProperty(name))
    return vane.FunctionExpression("image_attribute", _image_expression(image), as_expression(name))


def image_width(image: Any) -> vane.Expression:
    return vane.FunctionExpression("image_width", _image_expression(image))


def image_height(image: Any) -> vane.Expression:
    return vane.FunctionExpression("image_height", _image_expression(image))


def image_channel(image: Any) -> vane.Expression:
    return vane.FunctionExpression("image_channel", _image_expression(image))


def image_mode(image: Any) -> vane.Expression:
    """Return the mode code: L=1, LA=2, RGB=3, RGBA=4."""
    return vane.FunctionExpression("image_mode", _image_expression(image))
