# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Encoded IMAGEFILE metadata and Python decoding helpers."""

from __future__ import annotations

import contextlib
import importlib
import io
import threading
import warnings
from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import vane
from vane._expressions import as_expression
from vane._file import _positive_buffer_size

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage  # type: ignore[import-not-found]


DEFAULT_IMAGE_METADATA_BYTES = 1024 * 1024
DEFAULT_IMAGE_MAX_PIXELS = 100_000_000
DEFAULT_IMAGE_BUFFER_SIZE = 1024 * 1024
DEFAULT_IMAGE_MAX_INPUT_BYTES = 256 * 1024 * 1024
DEFAULT_IMAGE_MAX_DECODED_BYTES = 512 * 1024 * 1024
MAX_IMAGE_METADATA_BYTES = 64 * 1024 * 1024
_MAX_UBIGINT = (1 << 64) - 1

_EXPLICIT_IMAGE_MODES = frozenset({"1", "L", "LA", "P", "RGB", "RGBA", "CMYK", "YCbCr", "I", "F"})
_MIME_ALIASES = {
    "image/jpg": "image/jpeg",
    "image/pjpeg": "image/jpeg",
    "image/x-png": "image/png",
    "image/x-icon": "image/vnd.microsoft.icon",
}
_GENERIC_MIME_TYPES = frozenset({"application/octet-stream", "binary/octet-stream", "image/*"})
_PILLOW_LIMIT_LOCK = threading.RLock()


@dataclass(frozen=True, slots=True)
class ImageMetadata:
    """Header metadata for an encoded :class:`vane.ImageFile`."""

    width: int
    height: int
    format: str
    mode: str


class ImageFileError(RuntimeError):
    """Base class for failures caused by encoded image content."""


class ImageFileFormatError(ImageFileError):
    """The logical FILE view is not a supported, internally consistent image."""


class ImageFileLimitError(ImageFileError):
    """An ImageFile operation exceeded an explicit resource limit."""


class _ImageReaderError(Exception):
    def __init__(self, cause: Exception) -> None:
        super().__init__(str(cause))
        self.cause = cause


class _ImageReaderProxy:
    """Keep reader failures distinct from Pillow's OSError media failures."""

    def __init__(self, reader: vane.VaneFileReader) -> None:
        self._reader = reader

    def read(self, size: int = -1, /) -> bytes:
        try:
            return self._reader.read(size)
        except Exception as error:
            raise _ImageReaderError(error) from error

    def readline(self, size: int = -1, /) -> bytes:
        try:
            return self._reader.readline(size)
        except Exception as error:
            raise _ImageReaderError(error) from error

    def seek(self, offset: int, whence: int = 0, /) -> int:
        try:
            return self._reader.seek(offset, whence)
        except Exception as error:
            raise _ImageReaderError(error) from error

    def tell(self) -> int:
        try:
            return self._reader.tell()
        except Exception as error:
            raise _ImageReaderError(error) from error


class _MetadataBuffer(io.BytesIO):
    """Record whether a parser actually attempted to read past the budget."""

    def __init__(self, data: bytes, *, truncated: bool) -> None:
        super().__init__(data)
        self._length = len(data)
        self._truncated = truncated
        self.budget_exhausted = False

    def read(self, size: int | None = -1, /) -> bytes:
        result = super().read(size)
        if self._truncated and self.tell() >= self._length:
            self.budget_exhausted = True
        return result

    def readline(self, size: int | None = -1, /) -> bytes:
        result = super().readline(size)
        if self._truncated and self.tell() >= self._length:
            self.budget_exhausted = True
        return result


def _load_pillow() -> tuple[Any, type[Exception]]:
    try:
        image_module = importlib.import_module("PIL.Image")
        unidentified_error = importlib.import_module("PIL").UnidentifiedImageError
    except (ImportError, AttributeError) as error:
        raise ImportError(
            "ImageFile metadata and decoding require the 'pillow' package. Please `pip install 'vane-ai[image]'`."
        ) from error
    return image_module, unidentified_error


@contextlib.contextmanager
def _open_image_with_limit(image_module: Any, stream: Any, *, max_pixels: int) -> Iterator[Any]:
    # Pillow consults a process-global limit while opening and, for some
    # formats, again while loading pixels. Serialize the complete operation,
    # loosen the global cap only as far as this call requires, and restore it
    # afterwards. Vane's per-call validation remains the authoritative limit.
    with _PILLOW_LIMIT_LOCK:
        previous_limit = image_module.MAX_IMAGE_PIXELS
        pillow_limit = previous_limit
        # Pillow raises only above twice MAX_IMAGE_PIXELS; the warning between
        # the two thresholds is redundant with Vane's exact validation below.
        minimum_pillow_limit = (max_pixels + 1) // 2
        if previous_limit is not None and previous_limit < minimum_pillow_limit:
            pillow_limit = minimum_pillow_limit
        image_module.MAX_IMAGE_PIXELS = pillow_limit
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", image_module.DecompressionBombWarning)
                with image_module.open(stream) as image:
                    yield image
        finally:
            image_module.MAX_IMAGE_PIXELS = previous_limit


def _positive_limit(value: object, *, name: str, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be int, not {type(value).__name__!r}")
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be at most {maximum}")
    return value


def _limit_expression(value: int | vane.Expression, *, name: str, maximum: int | None = None) -> vane.Expression:
    if isinstance(value, vane.Expression):
        return value
    return as_expression(_positive_limit(value, name=name, maximum=maximum))


def _canonical_mime_type(value: str) -> str:
    normalized = value.partition(";")[0].strip().lower()
    return _MIME_ALIASES.get(normalized, normalized)


def _validate_content_type(content_type: str | None, detected_mime_type: str | None) -> None:
    if content_type is None:
        return
    declared = _canonical_mime_type(content_type)
    if declared in _GENERIC_MIME_TYPES:
        return
    if not declared.startswith("image/"):
        raise ImageFileFormatError(f"IMAGEFILE content_type {content_type!r} contradicts the decoded image content")
    if detected_mime_type is None:
        return
    detected = _canonical_mime_type(detected_mime_type)
    if declared != detected:
        raise ImageFileFormatError(
            f"IMAGEFILE content_type {content_type!r} contradicts detected MIME type {detected_mime_type!r}"
        )


def _validate_dimensions(width: int, height: int, max_pixels: int) -> None:
    if width <= 0 or height <= 0:
        raise ImageFileFormatError(f"image dimensions must be positive, found {width}x{height}")
    pixels = width * height
    if pixels > max_pixels:
        raise ImageFileLimitError(f"image contains {pixels} pixels, exceeding max_pixels={max_pixels}")


def _metadata_from_image(
    image: Any,
    image_module: Any,
    *,
    max_pixels: int,
    content_type: str | None,
) -> ImageMetadata:
    width, height = image.size
    width = int(width)
    height = int(height)
    _validate_dimensions(width, height, max_pixels)

    image_format = image.format
    if not isinstance(image_format, str) or not image_format:
        raise ImageFileFormatError("image decoder did not report an encoded format")
    mode = image.mode
    if not isinstance(mode, str) or not mode:
        raise ImageFileFormatError("image decoder did not report a pixel mode")
    detected_mime_type = image_module.MIME.get(image_format.upper())
    _validate_content_type(content_type, detected_mime_type)
    return ImageMetadata(width=width, height=height, format=image_format.upper(), mode=mode)


def _classified_image_errors(unidentified_error: type[Exception]) -> tuple[type[Exception], ...]:
    return (unidentified_error, OSError, SyntaxError, ValueError, EOFError)


def _probe_image_metadata(
    data: bytes,
    max_pixels: int,
    truncated: bool,
    content_type: str | None,
    max_bytes: int,
) -> tuple[int, int, str, str]:
    """Bounded helper called by the native SQL scalar function."""
    image_module, unidentified_error = _load_pillow()
    stream = _MetadataBuffer(data, truncated=truncated)
    try:
        with _open_image_with_limit(image_module, stream, max_pixels=max_pixels) as image:
            metadata = _metadata_from_image(
                image,
                image_module,
                max_pixels=max_pixels,
                content_type=content_type,
            )
    except ImageFileError:
        raise
    except image_module.DecompressionBombError as error:
        raise ImageFileLimitError(f"image dimensions exceed max_pixels={max_pixels}") from error
    except _classified_image_errors(unidentified_error) as error:
        if stream.budget_exhausted:
            raise ImageFileLimitError(f"image metadata requires more than max_bytes={max_bytes}") from error
        raise ImageFileFormatError("logical FILE view is not a supported encoded image") from error
    return metadata.width, metadata.height, metadata.format, metadata.mode


def _image_file_metadata_value(
    value: vane.ImageFile,
    *,
    max_bytes: int = DEFAULT_IMAGE_METADATA_BYTES,
    max_pixels: int = DEFAULT_IMAGE_MAX_PIXELS,
    connection: vane.DuckDBPyConnection | None = None,
) -> ImageMetadata:
    normalized_max_bytes = _positive_limit(max_bytes, name="max_bytes", maximum=MAX_IMAGE_METADATA_BYTES)
    normalized_max_pixels = _positive_limit(max_pixels, name="max_pixels", maximum=_MAX_UBIGINT)
    _load_pillow()
    with value.open(connection=connection) as reader:
        logical_size = reader.size()
        read_size = min(logical_size, normalized_max_bytes)
        data = reader.read(read_size)
    fields = _probe_image_metadata(
        data,
        normalized_max_pixels,
        logical_size > read_size,
        value.content_type,
        normalized_max_bytes,
    )
    return ImageMetadata(*fields)


def _decoded_bytes(image: Any, output_mode: str) -> int:
    if output_mode in {"1", "L", "P"}:
        bytes_per_pixel = 1
    elif output_mode.startswith("I;16"):
        bytes_per_pixel = 2
    elif output_mode in {"LA", "RGB", "RGBA", "CMYK", "YCbCr", "I", "F"}:
        bytes_per_pixel = 4
    else:
        bytes_per_pixel = max(4, len(image.getbands()))
    return int(image.width) * int(image.height) * bytes_per_pixel


def _validate_decode_mode(mode: object) -> str | None:
    if mode is None:
        return None
    if not isinstance(mode, str):
        raise TypeError(f"mode must be str or None, not {type(mode).__name__!r}")
    if mode not in _EXPLICIT_IMAGE_MODES:
        choices = ", ".join(sorted(_EXPLICIT_IMAGE_MODES))
        raise ValueError(f"unsupported image decode mode {mode!r}; expected one of: {choices}")
    return mode


def _decode_image_file(
    value: vane.ImageFile,
    mode: str | None = None,
    buffer_size: int = DEFAULT_IMAGE_BUFFER_SIZE,
    *,
    max_input_bytes: int = DEFAULT_IMAGE_MAX_INPUT_BYTES,
    max_pixels: int = DEFAULT_IMAGE_MAX_PIXELS,
    max_decoded_bytes: int = DEFAULT_IMAGE_MAX_DECODED_BYTES,
    connection: vane.DuckDBPyConnection | None = None,
) -> PILImage:
    normalized_mode = _validate_decode_mode(mode)
    normalized_buffer_size = _positive_buffer_size(buffer_size, none_default=None, name="buffer_size")
    normalized_max_input = _positive_limit(max_input_bytes, name="max_input_bytes", maximum=_MAX_UBIGINT)
    normalized_max_pixels = _positive_limit(max_pixels, name="max_pixels", maximum=_MAX_UBIGINT)
    normalized_max_decoded = _positive_limit(max_decoded_bytes, name="max_decoded_bytes", maximum=_MAX_UBIGINT)
    image_module, unidentified_error = _load_pillow()

    try:
        with value.open(buffer_size=normalized_buffer_size, connection=connection) as reader:
            input_size = reader.size()
            if input_size > normalized_max_input:
                raise ImageFileLimitError(
                    f"encoded image contains {input_size} bytes, exceeding max_input_bytes={normalized_max_input}"
                )
            with _open_image_with_limit(
                image_module,
                _ImageReaderProxy(reader),
                max_pixels=normalized_max_pixels,
            ) as source:
                metadata = _metadata_from_image(
                    source,
                    image_module,
                    max_pixels=normalized_max_pixels,
                    content_type=value.content_type,
                )
                output_mode = normalized_mode or metadata.mode
                source_bytes = _decoded_bytes(source, metadata.mode)
                output_bytes = _decoded_bytes(source, output_mode)
                decoded_working_bytes = source_bytes + output_bytes
                if decoded_working_bytes > normalized_max_decoded:
                    raise ImageFileLimitError(
                        f"image decode requires up to {decoded_working_bytes} bytes, "
                        f"exceeding max_decoded_bytes={normalized_max_decoded}"
                    )

                if normalized_mode is not None and normalized_mode != source.mode:
                    converted = source.convert(normalized_mode)
                    try:
                        converted.load()
                    except BaseException:
                        converted.close()
                        raise
                    return converted
                source.load()
                return source.copy()
    except _ImageReaderError as error:
        raise error.cause.with_traceback(error.cause.__traceback__)
    except ImageFileError:
        raise
    except image_module.DecompressionBombError as error:
        raise ImageFileLimitError(f"image dimensions exceed max_pixels={normalized_max_pixels}") from error
    except _classified_image_errors(unidentified_error) as error:
        raise ImageFileFormatError("logical FILE view is not a supported encoded image") from error


def image_file_metadata(
    value: vane.ImageFile | vane.Expression,
    *,
    max_bytes: int | vane.Expression = DEFAULT_IMAGE_METADATA_BYTES,
    max_pixels: int | vane.Expression = DEFAULT_IMAGE_MAX_PIXELS,
) -> vane.Expression:
    """Inspect bounded encoded IMAGEFILE headers without decoding pixels."""
    return vane.FunctionExpression(
        "image_file_metadata",
        as_expression(value),
        _limit_expression(max_bytes, name="max_bytes", maximum=MAX_IMAGE_METADATA_BYTES),
        _limit_expression(max_pixels, name="max_pixels", maximum=_MAX_UBIGINT),
    )


__all__ = [
    "ImageFileError",
    "ImageFileFormatError",
    "ImageFileLimitError",
    "ImageMetadata",
    "image_file_metadata",
]
