# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Explicit Python Image codec and hashing implementations."""

from __future__ import annotations

import importlib
import io
import math
from typing import Any, Callable

import numpy as np

from vane._image import _MODE_CHANNELS, _MODE_DTYPES, _array_mode, _validate_pixels
from vane._image_operators import _convert_image

_MAX_BYTES = 256 * 1024 * 1024
_MAX_PIXELS = 100_000_000


class ImageDecodeContentError(ValueError):
    """Malformed or unsupported encoded content; eligible for on_error='null'."""


def _tiff_codec_content_error(error: RuntimeError) -> bool:
    """Recognize codec corruption without swallowing wrapped allocation errors."""
    if type(error).__module__.partition(".")[0] != "imagecodecs":
        return False
    name = type(error).__name__
    message = str(error)
    # These codecs expose status names in the message, but no numeric attribute.
    statuses = {
        "ZlibError": {"Z_DATA_ERROR", "Z_BUF_ERROR", "Z_NEED_DICT"},
        "DeflateError": {"LIBDEFLATE_BAD_DATA", "LIBDEFLATE_SHORT_OUTPUT", "LIBDEFLATE_INSUFFICIENT_SPACE"},
        "ImcdError": {
            "IMCD_INPUT_CORRUPT",
            "IMCD_OUTPUT_TOO_SMALL",
            "IMCD_LZW_INVALID",
            "IMCD_LZW_NOTIMPLEMENTED",
            "IMCD_LZW_BUFFER_TOO_SMALL",
            "IMCD_LZW_TABLE_TOO_SMALL",
            "IMCD_LZW_CORRUPT",
        },
    }
    if name in statuses:
        return message.rpartition(" returned ")[2].strip("'") in statuses[name]
    if name in ("Jpeg8Error", "Jpeg12Error"):
        # libjpeg reports formatted text, including its own allocation failures.
        return not any(
            token in message.lower()
            for token in (
                "memory",
                "alloc",
                "backing store",
                "temporary file",
                "version",
                "internal",
            )
        )
    return False


class _CodecBuffer(io.BytesIO):
    def __init__(self, limit: int, check: Callable[[], None]) -> None:
        super().__init__()
        self.limit, self.check = limit, check

    def write(self, value: Any) -> int:
        self.check()
        if len(value) > self.limit - self.tell():
            raise OverflowError("Image encoding exceeds its output byte limit")
        return super().write(value)

    def seek(self, offset: int, whence: int = 0) -> int:
        self.check()
        target = offset if whence == 0 else self.tell() + offset if whence == 1 else len(self.getbuffer()) + offset
        if not 0 <= target <= self.limit:
            raise OverflowError("Image encoding seek exceeds its output byte limit")
        return super().seek(offset, whence)


def _check_shape(width: int, height: int, channels: int, itemsize: int, limit: int) -> None:
    if width <= 0 or height <= 0 or width > (1 << 32) - 1 or height > (1 << 32) - 1:
        raise ImageDecodeContentError("Invalid image dimensions")
    if width * height > _MAX_PIXELS or width * height * channels * itemsize > limit:
        raise OverflowError("Image decoding exceeds its pixel or byte limit")


def _decode_image_bytes(
    encoded: memoryview,
    mode: str | None,
    remaining: int,
    storage_width: int,
    check: Callable[[], None],
    *,
    max_pixels: int = _MAX_PIXELS,
    max_decoded_bytes: int = 512 * 1024 * 1024,
) -> tuple[np.ndarray, str]:
    PILImage = importlib.import_module("PIL.Image")
    UnidentifiedImageError = importlib.import_module("PIL").UnidentifiedImageError
    import vane
    from vane._image_file import ImageFileFormatError, ImageFileLimitError, _open_image_with_limit, _tiff_image_mode

    def check_decode(width: int, height: int, source_width: int, output_mode: str) -> None:
        if width * height > max_pixels:
            raise OverflowError("Image decoder exceeds max_pixels")
        channels = _MODE_CHANNELS[output_mode]
        pixel_width = _MODE_DTYPES[output_mode].itemsize
        working_width = source_width * 2 + channels * (storage_width + pixel_width)
        if width * height * working_width > max_decoded_bytes:
            raise OverflowError("Image decoder exceeds max_decoded_bytes")

    inferred: str | None = None
    check()
    if len(encoded) > _MAX_BYTES:
        raise OverflowError("Image decoding exceeds its input byte limit")
    signature = bytes(encoded[:4])
    is_tiff = signature in (b"II*\0", b"MM\0*", b"II+\0", b"MM\0+")
    try:
        with io.BytesIO(encoded) as source:
            if is_tiff:
                # TIFF float/RGB16 pixels cannot be represented by a Pillow RGB
                # image. Read their declared dtype before allocating any pixels.
                importlib.import_module("imagecodecs")  # Required compression dependency.
                tifffile = importlib.import_module("tifffile")

                with tifffile.TiffFile(source) as tiff:
                    try:
                        page = tiff.pages[0]
                    except IndexError as error:
                        raise ImageDecodeContentError("TIFF contains no image pages") from error
                    inferred = _tiff_image_mode(page)
                    width, height = page.imagewidth, page.imagelength
                    dtype = page.dtype.newbyteorder("=")
                    channels = page.samplesperpixel
                    check_decode(width, height, channels * dtype.itemsize, mode or inferred)
                    _check_shape(width, height, channels, dtype.itemsize, _MAX_BYTES)
                    _check_shape(width, height, _MODE_CHANNELS[mode or inferred], storage_width, remaining)
                    try:
                        pixels = page.asarray(maxworkers=1)
                    except RuntimeError as error:
                        if _tiff_codec_content_error(error):
                            raise ImageDecodeContentError(str(error)) from error
                        raise
                    if page.planarconfig == 2:
                        pixels = np.moveaxis(pixels, 0, -1)
                    if pixels.ndim == 2:
                        pixels = pixels[:, :, None]
                    pixels = np.ascontiguousarray(pixels, dtype=dtype)
                    if page.photometric == 0:
                        pixels = pixels.copy()
                        pixels[:, :, 0] = np.iinfo(dtype).max - pixels[:, :, 0]
            else:
                with _open_image_with_limit(PILImage, source, max_pixels=min(max_pixels, _MAX_PIXELS)) as probe:
                    if probe.format not in ("PNG", "JPEG", "GIF", "BMP"):
                        raise ImageDecodeContentError("Unsupported encoded image format")
                    width, height = probe.size
                    wide = probe.format == "PNG" and len(encoded) >= 26 and encoded[24] == 16
                    _check_shape(width, height, 4, 2 if wide else 1, _MAX_BYTES)
                    if wide:
                        imagecodecs = importlib.import_module("imagecodecs")

                        inferred = {0: "L16", 2: "RGB16", 4: "LA16", 6: "RGBA16"}.get(encoded[25])
                        if inferred is None:
                            raise ImageDecodeContentError("Invalid 16-bit PNG color type")
                        if "transparency" in probe.info and inferred in ("L16", "RGB16"):
                            inferred = "LA16" if inferred == "L16" else "RGBA16"
                        check_decode(width, height, 8, mode or inferred)
                        _check_shape(width, height, _MODE_CHANNELS[mode or inferred], storage_width, remaining)
                        shape = (height, width, _MODE_CHANNELS[inferred])
                        pixels = np.empty(shape if shape[2] > 1 else shape[:2], dtype=np.uint16)
                        try:
                            imagecodecs.png_decode(encoded, out=pixels)
                        except imagecodecs.PngError as exc:
                            raise ImageDecodeContentError(str(exc)) from exc
                        pixels = pixels.reshape(shape)
                    else:
                        inferred = "L" if probe.mode == "1" else probe.mode
                        if inferred not in ("L", "LA", "RGB", "RGBA"):
                            inferred = "RGBA" if "transparency" in probe.info or inferred == "P" else "RGB"
                        if "transparency" in probe.info and inferred in ("L", "RGB"):
                            inferred = "LA" if inferred == "L" else "RGBA"
                        check_decode(width, height, 4, mode or inferred)
                        _check_shape(width, height, _MODE_CHANNELS[mode or inferred], storage_width, remaining)
                        converted = probe.convert(inferred)
                        try:
                            pixels = np.asarray(converted).reshape(height, width, _MODE_CHANNELS[inferred]).copy()
                        finally:
                            converted.close()
    except (PILImage.DecompressionBombError, PILImage.DecompressionBombWarning, ImageFileLimitError) as exc:
        raise OverflowError(f"Image decoder exceeded max_pixels={min(max_pixels, _MAX_PIXELS)}") from exc
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError, EOFError, ImageFileFormatError) as exc:
        if isinstance(exc, OSError) and exc.errno is not None:
            raise
        if isinstance(exc, ImageDecodeContentError):
            raise
        raise ImageDecodeContentError(str(exc)) from exc
    check()
    try:
        inferred = _array_mode(pixels)
        _validate_pixels(pixels, inferred)
    except vane.InvalidInputException as exc:
        raise ImageDecodeContentError(str(exc)) from exc
    result_mode = inferred if mode is None else mode
    _check_shape(width, height, _MODE_CHANNELS[result_mode], storage_width, remaining)
    if result_mode != inferred:
        converted = np.empty((height, width, _MODE_CHANNELS[result_mode]), dtype=_MODE_DTYPES[result_mode])
        _convert_image(
            memoryview(pixels),
            width,
            height,
            pixels.shape[2],
            converted.shape[2],
            memoryview(converted),
            check,
            inferred,
            result_mode,
        )
        pixels = converted
    return pixels, result_mode


def _encode_image_bytes(
    pixels: memoryview, width: int, height: int, mode: str, image_format: str, limit: int, check: Callable[[], None]
) -> bytes:
    check()
    source = np.frombuffer(pixels, dtype=_MODE_DTYPES[mode]).reshape(height, width, _MODE_CHANNELS[mode])
    with _CodecBuffer(limit, check) as output:
        if image_format == "TIFF":
            tifffile = importlib.import_module("tifffile")

            tifffile.imwrite(
                output,
                source[:, :, 0] if source.shape[2] == 1 else source,
                photometric="minisblack" if source.shape[2] < 3 else "rgb",
                extrasamples=["unassalpha"] if source.shape[2] in (2, 4) else None,
                metadata=None,
                rowsperstrip=max(1, 65536 // (width * source.shape[2] * source.itemsize)),
            )
        elif image_format == "PNG" and source.dtype == np.uint16:
            imagecodecs = importlib.import_module("imagecodecs")

            # The codec writes into a preallocated bounded destination.
            raw = (width * source.shape[2] * 2 + 1) * height
            capacity = min(limit, raw + (raw >> 12) + (raw >> 14) + (raw >> 25) + 65536)
            buffer = np.empty(capacity, dtype=np.uint8)
            try:
                encoded = imagecodecs.png_encode(source[:, :, 0] if source.shape[2] == 1 else source, out=buffer)
            except imagecodecs.PngError as exc:
                raise OverflowError("PNG encoder could not fit its bounded output buffer") from exc
            output.write(encoded)
        else:
            PILImage = importlib.import_module("PIL.Image")

            image = PILImage.fromarray(source[:, :, 0] if source.shape[2] == 1 else source)
            try:
                options = {"quality": 95, "subsampling": 0} if image_format == "JPEG" else {}
                if image_format == "GIF":
                    quantized = image.quantize(
                        colors=256, method=PILImage.Quantize.MEDIANCUT, dither=PILImage.Dither.NONE
                    )
                    image.close()
                    image = quantized
                image.save(output, format=image_format, **options)
            finally:
                image.close()
        check()
        return output.getvalue()


def _gray(pixels: np.ndarray, mode: str, check: Callable[[], None]) -> np.ndarray:
    source = pixels.reshape(-1, pixels.shape[2])
    result: np.ndarray = np.empty(source.shape[0], dtype=np.uint8)
    maximum = 1.0 if pixels.dtype == np.float32 else float(np.iinfo(pixels.dtype).max)
    for begin in range(0, len(source), 16384):
        check()
        block = source[begin : begin + 16384].astype(np.float64) * (255 / maximum)
        gray = block[:, 0] if block.shape[1] < 3 else (299 * block[:, 0] + 587 * block[:, 1] + 114 * block[:, 2]) / 1000
        result[begin : begin + len(block)] = np.floor(np.clip(gray, 0, 255) + 0.5)
    return result.reshape(pixels.shape[:2])


def _resize_gray(source: np.ndarray, height: int, width: int, check: Callable[[], None]) -> np.ndarray:
    vertical = source.shape[0] / height >= source.shape[1] / width
    intermediate = source.shape[1] * height if vertical else source.shape[0] * width
    other = max(source.shape[1], height) if vertical else max(source.shape[0], width)
    if source.size + intermediate + height * width + 8 * other + 1024 * 1024 > _MAX_BYTES:
        raise OverflowError("Image hash sampling exceeds its scratch limit")

    def axis_pass(values: np.ndarray, axis: int, target: int) -> np.ndarray:
        length = values.shape[axis]
        if length == target:
            return values
        moved = np.moveaxis(values, axis, 0)
        result = np.empty((target, moved.shape[1]), dtype=np.uint8)
        ratio = length / target
        support = max(1.0, ratio)
        chunk = max(1, 16384 // moved.shape[1])
        for i in range(target):
            check()
            center = (i + 0.5) * ratio
            begin, end = max(0, math.ceil(center - support - 0.5)), min(length, math.floor(center + support - 0.5) + 1)
            total = np.zeros(moved.shape[1], dtype=np.float64)
            weight_sum = 0.0
            for start in range(begin, end, chunk):
                check()
                stop = min(start + chunk, end)
                weights = np.maximum(
                    0.0, 1.0 - np.abs((np.arange(start, stop, dtype=np.float64) + 0.5 - center) / support)
                )
                total += np.sum(moved[start:stop].astype(np.float64) * weights[:, None], axis=0)
                weight_sum += float(weights.sum())
            result[i] = np.floor(total / weight_sum + 0.5)
        return np.moveaxis(result, 0, axis)

    # Reduce the larger ratio first, keeping the intermediate no larger than
    # the source or target. Both passes round to an 8-bit grayscale sample.
    if vertical:
        return axis_pass(axis_pass(source, 0, height), 1, width)
    return axis_pass(axis_pass(source, 1, width), 0, height)


def _gray_hash(gray: np.ndarray, method: str, size: int, check: Callable[[], None]) -> np.ndarray:
    if method in ("ahash", "dhash", "dhash_vertical"):
        samples = _resize_gray(gray, size + (method == "dhash_vertical"), size + (method == "dhash"), check)
        if method == "dhash":
            return (samples[:, 1:] > samples[:, :-1]).reshape(-1)
        if method == "dhash_vertical":
            return (samples[1:] > samples[:-1]).reshape(-1)
        return (samples > samples.mean()).reshape(-1)
    if method == "whash":
        scale = max(size, 1 << (min(gray.shape).bit_length() - 1))
        samples = _resize_gray(gray, scale, scale, check)
        block = scale // size
        means: np.ndarray = np.empty((size, size), dtype=np.float64)
        for y in range(size):
            check()
            for x in range(size):
                means[y, x] = samples[y * block : (y + 1) * block, x * block : (x + 1) * block].mean()
        return (means > means.mean()).reshape(-1)
    n = size * 4
    samples = _resize_gray(gray, n, n, check).astype(np.float64)
    frequencies: np.ndarray = np.arange(size + 1, dtype=np.float64)[:, None]
    basis = np.cos(np.pi * frequencies * (2 * np.arange(n) + 1) / (2 * n))
    check()
    if method == "phash_simple":
        coefficients = samples[:size] @ basis[1 : size + 1].T
    else:
        coefficients = basis[:size] @ samples @ basis[:size].T
    # A declared quantization avoids unstable threshold decisions caused by
    # sub-micro-unit DCT roundoff across scalar and BLAS implementations.
    coefficients = np.floor(coefficients * 1e6 + 0.5) / 1e6
    threshold = coefficients.mean() if method == "phash_simple" else np.median(coefficients)
    return (coefficients > threshold).reshape(-1)


def _image_hash(
    pixels: memoryview,
    width: int,
    height: int,
    mode: str,
    method: str,
    size: int,
    binbits: int,
    segments: int,
    check: Callable[[], None],
) -> bytes:
    source = np.frombuffer(pixels, dtype=_MODE_DTYPES[mode]).reshape(height, width, _MODE_CHANNELS[mode])
    if method == "colorhash":
        counts: np.ndarray = np.zeros(14, dtype=np.int64)
        maximum = 1.0 if source.dtype == np.float32 else float(np.iinfo(source.dtype).max)
        flat = source.reshape(-1, source.shape[2])
        for begin in range(0, len(flat), 16384):
            check()
            values = flat[begin : begin + 16384].astype(np.float64)
            rgb = values[:, :3] if values.shape[1] >= 3 else np.repeat(values[:, :1], 3, axis=1)
            rgb = np.floor(np.clip(rgb * (255 / maximum), 0, 255) + 0.5).astype(np.int64)
            red, green, blue = rgb.T
            high, low = rgb.max(axis=1), rgb.min(axis=1)
            delta = high - low
            saturation = 255 * delta // np.maximum(high, 1)
            divisor = np.maximum(6 * delta, 1)
            numerator = np.where(
                high == red,
                (green - blue) * 255,
                np.where(high == green, 85 * divisor + (blue - red) * 255, 170 * divisor + (red - green) * 255),
            )
            hue = np.where(delta == 0, 0, (numerator // divisor) % 255)
            intensity = (299 * red + 587 * green + 114 * blue + 500) // 1000
            black = intensity < 32
            gray = ~black & (saturation < 85)
            color = ~black & ~gray
            bins = np.minimum(5, hue * 6 // 255)
            counts[0] += black.sum()
            counts[1] += gray.sum()
            counts[2:8] += np.bincount(bins[color & (saturation <= 170)], minlength=6)
            counts[8:14] += np.bincount(bins[color & (saturation > 170)], minlength=6)
        levels = 1 << binbits
        denominators = np.array([width * height] * 2 + [max(1, int(counts[2:].sum()))] * 12)
        values = np.minimum(levels - 1, counts * levels // denominators)
        bits = np.array([((int(value) >> k) & 1) != 0 for value in values for k in range(binbits - 1, -1, -1)])
    else:
        gray = _gray(source, mode, check)
        if method == "crop_resistant":
            if width < segments or height < segments:
                raise ValueError("crop_resistant requires at least one pixel per grid segment")
            bits = np.concatenate(
                [
                    _gray_hash(
                        gray[
                            y * height // segments : (y + 1) * height // segments,
                            x * width // segments : (x + 1) * width // segments,
                        ],
                        "phash",
                        size,
                        check,
                    )
                    for y in range(segments)
                    for x in range(segments)
                ]
            )
        else:
            bits = _gray_hash(gray, method, size, check)
    check()
    return np.packbits(bits, bitorder="big").tobytes()
