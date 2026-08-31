# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Encoded IMAGEFILE metadata and Python decoding helpers."""

from __future__ import annotations

import contextlib
import contextvars
import functools
import importlib
import io
import threading
from collections.abc import Callable, Iterator
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
    "image/j2k": "image/j2c",
    "image/jpc": "image/j2c",
    "image/jpg": "image/jpeg",
    "image/pjpeg": "image/jpeg",
    "image/x-portable-greymap": "image/x-portable-graymap",
    "image/x-png": "image/png",
    "image/x-icon": "image/vnd.microsoft.icon",
}
_GENERIC_MIME_TYPES = frozenset({"application/octet-stream", "binary/octet-stream", "image/*"})
_PILLOW_PIXEL_LIMIT = contextvars.ContextVar[int | None]("vane_image_file_max_pixels", default=None)
_PILLOW_HOOK_LOCK = threading.Lock()


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


@dataclass(slots=True)
class _PillowLimitHook:
    image_module: Any
    original: Callable[[tuple[int, int]], None]
    wrapper: Callable[[tuple[int, int]], None]
    users: int = 0


_PILLOW_HOOK: _PillowLimitHook | None = None


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


def _pillow_limit_wrapper(
    original: Callable[[tuple[int, int]], None],
) -> Callable[[tuple[int, int]], None]:
    @functools.wraps(original)
    def check(size: tuple[int, int]) -> None:
        max_pixels = _PILLOW_PIXEL_LIMIT.get()
        if max_pixels is None:
            original(size)
            return
        pixels = max(1, int(size[0])) * max(1, int(size[1]))
        if pixels > max_pixels:
            raise ImageFileLimitError(f"image contains {pixels} pixels, exceeding max_pixels={max_pixels}")

    return check


@contextlib.contextmanager
def _pillow_pixel_limit(image_module: Any, max_pixels: int) -> Iterator[None]:
    """Scope Pillow's process-global check through a context-local limit."""
    global _PILLOW_HOOK

    token = _PILLOW_PIXEL_LIMIT.set(max_pixels)
    try:
        with _PILLOW_HOOK_LOCK:
            hook = _PILLOW_HOOK
            if hook is None:
                original = image_module._decompression_bomb_check
                wrapper = _pillow_limit_wrapper(original)
                hook = _PillowLimitHook(image_module, original, wrapper)
                image_module._decompression_bomb_check = wrapper
                _PILLOW_HOOK = hook
            elif hook.image_module is not image_module or image_module._decompression_bomb_check is not hook.wrapper:
                raise RuntimeError("Pillow decompression check changed during an ImageFile operation")
            hook.users += 1
        try:
            yield
        finally:
            with _PILLOW_HOOK_LOCK:
                hook.users -= 1
                if hook.users == 0:
                    if hook.image_module._decompression_bomb_check is hook.wrapper:
                        hook.image_module._decompression_bomb_check = hook.original
                    if _PILLOW_HOOK is hook:
                        _PILLOW_HOOK = None
    finally:
        _PILLOW_PIXEL_LIMIT.reset(token)


@contextlib.contextmanager
def _open_image_with_limit(image_module: Any, stream: Any, *, max_pixels: int) -> Iterator[Any]:
    # Pillow has no per-open pixel limit and some plugins repeat its private
    # check while loading. The scoped hook delegates unchanged outside this
    # context, so unrelated threads retain their own global Pillow policy.
    with _pillow_pixel_limit(image_module, max_pixels):
        with image_module.open(stream) as image:
            yield image


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


def _validate_content_type(
    content_type: str | None,
    detected_mime_type: str | None,
    compatible_mime_types: frozenset[str],
) -> None:
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
    if declared != detected and declared not in compatible_mime_types:
        raise ImageFileFormatError(
            f"IMAGEFILE content_type {content_type!r} contradicts detected MIME type {detected_mime_type!r}"
        )


def _detected_image_mime_types(image: Any) -> tuple[str | None, frozenset[str]]:
    detected_mime_type = image.get_format_mimetype()
    compatible_mime_types: set[str] = set()

    if image.format == "JPEG2000" and getattr(image, "codec", None) == "j2k":
        # Pillow uses its JPEG2000-wide image/jp2 registry entry for raw J2K
        # codestreams, whose registered media type is image/j2c.
        detected_mime_type = "image/j2c"
    elif image.format == "PPM":
        # Pillow reports the precise PBM/PGM/PPM subtype when available. The
        # portable-anymap type remains a compatible family-level declaration.
        compatible_mime_types.add("image/x-portable-anymap")
        if image.mode == "F":
            compatible_mime_types.add("image/x-portable-floatmap")

    if detected_mime_type is not None:
        compatible_mime_types.add(_canonical_mime_type(detected_mime_type))
    return detected_mime_type, frozenset(compatible_mime_types)


def _validate_dimensions(width: int, height: int, max_pixels: int) -> None:
    if width <= 0 or height <= 0:
        raise ImageFileFormatError(f"image dimensions must be positive, found {width}x{height}")
    pixels = width * height
    if pixels > max_pixels:
        raise ImageFileLimitError(f"image contains {pixels} pixels, exceeding max_pixels={max_pixels}")


def _metadata_from_image(
    image: Any,
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
    detected_mime_type, compatible_mime_types = _detected_image_mime_types(image)
    _validate_content_type(content_type, detected_mime_type, compatible_mime_types)
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
