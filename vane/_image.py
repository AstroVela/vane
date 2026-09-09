# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Decoded UInt8, UInt16 and Float32 HWC Image values and Arrow transport."""

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

Image: TypeAlias = npt.NDArray[np.uint8] | npt.NDArray[np.uint16] | npt.NDArray[np.float32]


class _ImageStringEnum(str, Enum):
    def __str__(self) -> str:
        return str(self.value)


class ImageMode(_ImageStringEnum):
    L = "L"
    LA = "LA"
    RGB = "RGB"
    RGBA = "RGBA"
    L16 = "L16"
    LA16 = "LA16"
    RGB16 = "RGB16"
    RGBA16 = "RGBA16"
    RGB32F = "RGB32F"
    RGBA32F = "RGBA32F"


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


_MODE_CODES = {str(mode): code for code, mode in enumerate(ImageMode, 1)}
_MODE_CHANNELS = {mode: ((code - 1) % 4 + 1 if code <= 8 else code - 6) for mode, code in _MODE_CODES.items()}
_MODE_DTYPES: dict[str, np.dtype[Any]] = {
    mode: np.dtype("uint8" if code <= 4 else "uint16" if code <= 8 else "float32") for mode, code in _MODE_CODES.items()
}


def _pixel_dtype(mode: str | None) -> np.dtype[Any]:
    return np.dtype("float32") if mode is None else _MODE_DTYPES[mode]


def _array_mode(value: np.ndarray) -> str:
    candidates = [
        mode for mode in _MODE_CODES if _MODE_DTYPES[mode] == value.dtype and _MODE_CHANNELS[mode] == value.shape[2]
    ]
    if not candidates:
        raise vane.InvalidInputException(
            "Image requires UInt8/UInt16 with 1 to 4 channels, or Float32 with 3 or 4 channels"
        )
    return candidates[0]


def _validate_pixels(pixels: np.ndarray, mode: str) -> None:
    if not np.isfinite(pixels).all():
        raise vane.InvalidInputException("Image pixels must be finite")
    dtype = _MODE_DTYPES[mode]
    if dtype != np.float32 and (
        np.any(pixels < 0) or np.any(pixels > np.iinfo(dtype).max) or np.any(pixels != np.floor(pixels))
    ):
        raise vane.InvalidInputException(f"Image pixels are not exactly representable in mode {mode}")


_MODE_NAMES = {code: name for name, code in _MODE_CODES.items()}
_IMAGE_FIELDS = ("data", "channel", "height", "width", "mode")
_EXTENSION_NAME = "vane.image"


def _dynamic_storage(mode: str | None) -> pa.DataType:
    return pa.struct(
        [
            ("data", pa.list_(pa.from_numpy_dtype(_pixel_dtype(mode)))),
            ("channel", pa.uint16()),
            ("height", pa.uint32()),
            ("width", pa.uint32()),
            ("mode", pa.uint8()),
        ]
    )


_DYNAMIC_STORAGE = _dynamic_storage(None)


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
        storage = _dynamic_storage(mode)
        if height is not None:
            if mode is None or type(height) is not int or type(width) is not int or height <= 0 or width <= 0:
                raise ValueError("Fixed Image requires a mode and positive integer dimensions")
            size = height * width * _MODE_CHANNELS[mode]
            if size > (1 << 31) - 1:
                raise ValueError("Fixed Image cannot exceed 2147483647 pixel values")
            storage = pa.list_(pa.from_numpy_dtype(_pixel_dtype(mode)), size)
        super().__init__(storage, _EXTENSION_NAME)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _ImageArrowType):
            return NotImplemented
        # Arrow's native schema/concatenation checks also call this method.
        # Equal UInt8 storage sizes do not imply equal HWC layouts or modes.
        return (
            type(self) is type(other)
            and self.storage_type == other.storage_type
            and (self.mode, self.height, self.width) == (other.mode, other.height, other.width)
        )

    def __ne__(self, other: object) -> bool:
        # PyArrow defines a separate inequality slot; inheriting it would
        # still compare only the extension name and physical storage.
        equal = self.__eq__(other)
        if equal is NotImplemented:
            return NotImplemented
        return not equal

    def __hash__(self) -> int:
        return hash((type(self), self.storage_type, self.mode, self.height, self.width))

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
        mode = "L16" if value.mode in ("I;16", "I;16L", "I;16B") else value.mode
        if mode not in _MODE_CODES:
            raise vane.InvalidInputException(f"Unsupported Image mode {mode!r}")
        value = np.asarray(value).astype(_MODE_DTYPES[mode], copy=False)
        if _MODE_CHANNELS[mode] == 1:
            value = value[:, :, np.newaxis]
    if not isinstance(value, np.ndarray) or isinstance(value, np.ma.MaskedArray):
        raise vane.InvalidInputException("Image input must be an HWC ndarray or PIL.Image.Image")
    if value.ndim != 3 or not 1 <= value.shape[2] <= 4:
        raise vane.InvalidInputException("Image requires HWC pixels with 1 to 4 channels")
    height, width, channels = value.shape
    mode = _array_mode(value) if mode is None else mode
    if value.dtype != _MODE_DTYPES[mode]:
        raise vane.InvalidInputException("Image pixel dtype does not match its mode")
    _validate_pixels(value, mode)
    _validate_layout(dtype, width, height, mode)
    # Arrow's typed list builder consumes NumPy buffers directly. Expanding a
    # 4K image into Python list entries would multiply its memory footprint.
    pixels: np.ndarray = value.ravel(order="C").astype(_pixel_dtype(dtype.image_mode), copy=False)
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
        if stored_mode is None or value["channel"] != _MODE_CHANNELS[stored_mode]:
            raise vane.InvalidInputException("Image mode and channel count do not match")
        mode = stored_mode
        height, width, pixels = value["height"], value["width"], value["data"]
    _validate_layout(dtype, width, height, mode)
    channels = _MODE_CHANNELS[mode]
    if len(pixels) != height * width * channels or any(pixel is None for pixel in pixels):
        raise vane.InvalidInputException("Image pixels must be non-NULL values matching its HWC shape")
    array = np.asarray(pixels)
    _validate_pixels(array, mode)
    return array.astype(_MODE_DTYPES[mode]).reshape(height, width, channels)


def _image_arrow_scalar_to_numpy(value: Any, dtype: Any) -> Image:
    """Copy a validated Image scalar directly from Arrow's typed pixel buffer."""
    storage = value.value if isinstance(value, pa.ExtensionScalar) else value
    if dtype.is_fixed_shape_image():
        height, width = dtype.shape
        channels = _MODE_CHANNELS[str(dtype.image_mode)]
        mode = str(dtype.image_mode)
        pixels = storage.values
    else:
        height, width, channels = (storage[name].as_py() for name in ("height", "width", "channel"))
        pixels = storage["data"].values
        mode = _MODE_NAMES[storage["mode"].as_py()]
    return pixels.to_numpy().astype(_MODE_DTYPES[mode]).reshape(height, width, channels)


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


def image_to_tensor(image: Any) -> vane.Expression:
    """Convert Image pixels to an HWC Tensor without changing pixel values.

    Known modes preserve UInt8, UInt16 or Float32. Generic Image uses Float32,
    which exactly represents every supported integer pixel value.

    Fixed Images produce fixed shape Tensors. Dynamic Images retain their
    known channel count, with variable height and width. NULL stays NULL.
    The base C++ conversion shares pixel buffers and requires no image backend.
    """
    return vane.FunctionExpression("image_to_tensor", _image_expression(image))
