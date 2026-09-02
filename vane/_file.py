# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Python value, reader, expression, and discovery facade for FILE."""

from __future__ import annotations

import io
import shutil
import sys
import tempfile
from typing import TYPE_CHECKING, Any, BinaryIO, Literal

import vane
from vane._expressions import as_expression

if TYPE_CHECKING:
    from vane._native import _VaneFileReaderHandle


_DEFAULT_FILE_BUFFER_SIZE = 1024 * 1024
_DEFAULT_TEMPFILE_BUFFER_SIZE = 1024 * 1024


class VaneFileReader(io.RawIOBase):
    """A read-only, seekable view over a :class:`vane.File`."""

    def __init__(self, inner: _VaneFileReaderHandle) -> None:
        super().__init__()
        self._inner = inner

    def read(self, size: int = -1, /) -> bytes:
        return self._inner._read(size)

    def readinto(self, buffer: Any, /) -> int:
        return self._readinto_from(buffer, self._inner._read)

    def _read_and_check_interrupted(self, size: int = -1, /) -> bytes:
        return self._inner._read_and_check_interrupted(size)

    def _readinto_and_check_interrupted(self, buffer: Any, /) -> int:
        return self._readinto_from(buffer, self._inner._read_and_check_interrupted)

    @staticmethod
    def _readinto_from(buffer: Any, read: Any) -> int:
        try:
            view = memoryview(buffer)
        except TypeError as error:
            raise TypeError("readinto() argument must be a writable bytes-like object") from error
        byte_view: memoryview | None = None
        try:
            if view.readonly:
                raise TypeError("readinto() argument must be a writable bytes-like object")
            try:
                byte_view = view.cast("B")
            except TypeError as error:
                raise TypeError("readinto() argument must be a contiguous writable bytes-like object") from error
            data = read(byte_view.nbytes)
            byte_view[: len(data)] = data
            return len(data)
        finally:
            if byte_view is not None:
                byte_view.release()
            view.release()

    def write(self, _buffer: Any, /) -> int:
        self._require_open()
        raise io.UnsupportedOperation("VaneFileReader is not writable")

    def seek(self, offset: int, whence: int = io.SEEK_SET, /) -> int:
        return self._inner._seek(offset, whence)

    def tell(self) -> int:
        return self._inner._tell()

    def size(self) -> int:
        """Return the size of this FILE's logical byte view."""
        return self._inner._size()

    def _check_interrupted(self) -> None:
        self._inner._check_interrupted()

    def guess_mime_type(self) -> str | None:
        """Inspect bounded bytes without changing the current stream position."""
        return self._inner._guess_mime_type()

    def close(self) -> None:
        try:
            self._inner._close()
        finally:
            super().close()

    def _close_and_check_interrupted(self) -> None:
        try:
            self._inner._close_and_check_interrupted()
        finally:
            super().close()

    @property
    def closed(self) -> bool:
        return self._inner._closed

    def _require_open(self) -> None:
        if self.closed:
            raise ValueError("I/O operation on closed VaneFileReader")

    def readable(self) -> bool:
        self._require_open()
        return True

    def writable(self) -> bool:
        self._require_open()
        return False

    def seekable(self) -> bool:
        self._require_open()
        return True

    def isatty(self) -> bool:
        self._require_open()
        return False

    def __enter__(self) -> VaneFileReader:
        self._require_open()
        return self

    def __exit__(self, exc_type: object, _exc_value: object, _traceback: object) -> None:
        if exc_type is None:
            self.close()
            return
        try:
            self.close()
        except BaseException:
            # Cleanup must not replace an exception already escaping the reader
            # body, including control-flow exceptions injected into a generator.
            pass

    def __str__(self) -> str:
        return str(self._inner)

    def __repr__(self) -> str:
        return f"VaneFileReader(url={str(self._inner)!r}, closed={self.closed})"


def _positive_buffer_size(value: object, *, none_default: int | None, name: str) -> int:
    if value is None:
        if none_default is not None:
            return none_default
        raise TypeError(f"{name} must be int, not 'NoneType'")
    if isinstance(value, bool) or not isinstance(value, int):
        expected_type = "int or None" if none_default is not None else "int"
        raise TypeError(f"{name} must be {expected_type}, not {type(value).__name__!r}")
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    if value > sys.maxsize:
        raise OverflowError(f"{name} must fit in Py_ssize_t")
    return value


def _file_open(
    self: vane.File,
    buffer_size: int | None = None,
    *,
    connection: vane.DuckDBPyConnection | None = None,
) -> VaneFileReader:
    from vane._native import _open_file_reader

    normalized_size = _positive_buffer_size(
        buffer_size,
        none_default=_DEFAULT_FILE_BUFFER_SIZE,
        name="buffer_size",
    )
    return VaneFileReader(_open_file_reader(self, buffer_size=normalized_size, connection=connection))


def _file_to_tempfile(
    self: vane.File,
    buffer_size: int = _DEFAULT_TEMPFILE_BUFFER_SIZE,
    *,
    connection: vane.DuckDBPyConnection | None = None,
) -> BinaryIO:
    normalized_size = _positive_buffer_size(
        buffer_size,
        none_default=None,
        name="buffer_size",
    )
    temporary = tempfile.TemporaryFile(mode="w+b", prefix="vane_")
    try:
        with _file_open(self, normalized_size, connection=connection) as source:
            shutil.copyfileobj(source, temporary, length=normalized_size)
        temporary.seek(0)
        return temporary
    except BaseException:
        temporary.close()
        raise


def open_file(
    url: str,
    mode: Literal["r", "rt", "rb"] = "r",
    buffering: int = -1,
    encoding: str | None = None,
    errors: str | None = None,
    newline: str | None = None,
    *,
    connection: vane.DuckDBPyConnection | None = None,
) -> io.IOBase:
    """Open a URL as a standard read-only binary or text stream."""
    if not isinstance(mode, str):
        raise TypeError(f"mode must be str, not {type(mode).__name__!r}")
    if mode not in ("r", "rt", "rb"):
        raise ValueError(f"invalid file open mode {mode!r}; only 'r', 'rt', and 'rb' are supported")
    if isinstance(buffering, bool) or not isinstance(buffering, int):
        raise TypeError(f"buffering must be int, not {type(buffering).__name__!r}")
    if buffering < -1:
        raise ValueError("buffering must be -1, 0, 1, or a positive integer")
    text_mode = mode in ("r", "rt")
    if text_mode and buffering == 0:
        raise ValueError("unbuffered text I/O is not supported")
    if not text_mode and any(option is not None for option in (encoding, errors, newline)):
        raise ValueError("binary mode does not accept encoding, errors, or newline")

    if buffering == 0:
        stream_buffer_size = 1
    elif buffering in (-1, 1):
        stream_buffer_size = io.DEFAULT_BUFFER_SIZE
    else:
        stream_buffer_size = buffering
    raw = _file_open(vane.File(url), stream_buffer_size, connection=connection)
    if not text_mode and buffering == 0:
        return raw

    try:
        buffered = io.BufferedReader(raw, stream_buffer_size)
    except BaseException:
        raw.close()
        raise
    if not text_mode:
        return buffered
    try:
        return io.TextIOWrapper(
            buffered,
            encoding=encoding,
            errors=errors,
            newline=newline,
            line_buffering=buffering == 1,
        )
    except BaseException:
        buffered.close()
        raise


def _file_function(name: str, *arguments: object) -> vane.Expression:
    return vane.FunctionExpression(name, *(as_expression(argument) for argument in arguments))


def file(
    url: str | vane.Expression,
    content_type: str | vane.Expression | None = None,
    position: int | vane.Expression | None = None,
    size: int | vane.Expression | None = None,
    checksum: str | vane.Expression | None = None,
) -> vane.Expression:
    """Construct a FILE expression without accessing the referenced resource."""
    return vane.FunctionExpression(
        "file",
        as_expression(url),
        as_expression(content_type),
        as_expression(position),
        as_expression(size),
        as_expression(checksum),
    )


def _media_file(
    name: str,
    value: str | vane.File | vane.Expression,
    verify: bool | vane.Expression,
) -> vane.Expression:
    arguments = [as_expression(value)]
    if verify is not False:
        arguments.append(as_expression(verify))
    return vane.FunctionExpression(name, *arguments)


def image_file(
    url_or_file: str | vane.File | vane.Expression,
    verify: bool | vane.Expression = False,
) -> vane.Expression:
    """Declare an IMAGEFILE expression, optionally verifying bounded content."""
    return _media_file("image_file", url_or_file, verify)


def audio_file(
    url_or_file: str | vane.File | vane.Expression,
    verify: bool | vane.Expression = False,
) -> vane.Expression:
    """Declare an AUDIOFILE expression, optionally verifying bounded content."""
    return _media_file("audio_file", url_or_file, verify)


def video_file(
    url_or_file: str | vane.File | vane.Expression,
    verify: bool | vane.Expression = False,
) -> vane.Expression:
    """Declare a VIDEOFILE expression, optionally verifying bounded content."""
    return _media_file("video_file", url_or_file, verify)


def to_file(path: str | vane.Expression) -> vane.Expression:
    """Convert a path to a FILE expression when the expression is executed."""
    return _file_function("to_file", path)


def try_to_file(path: str | vane.Expression) -> vane.Expression:
    """Convert a path to FILE, returning NULL for recoverable access failures."""
    return _file_function("try_to_file", path)


def file_enrich(value: vane.File | vane.Expression, fields: list[str] | vane.Expression) -> vane.Expression:
    """Enrich selected FILE fields when the expression is executed."""
    return _file_function("file_enrich", value, fields)


def file_path(value: vane.File | vane.Expression) -> vane.Expression:
    return _file_function("file_path", value)


def file_size(value: vane.File | vane.Expression) -> vane.Expression:
    return _file_function("file_size", value)


def file_exists(value: vane.File | vane.Expression) -> vane.Expression:
    return _file_function("file_exists", value)


def file_stat(value: vane.File | vane.Expression) -> vane.Expression:
    return _file_function("file_stat", value)


def file_mime_type(
    value: vane.File | vane.Expression,
    detect: str | vane.Expression = "metadata",
) -> vane.Expression:
    if isinstance(detect, str) and detect == "metadata":
        return _file_function("file_mime_type", value)
    return _file_function("file_mime_type", value, detect)


def guess_mime_type(value: bytes | vane.Expression) -> vane.Expression:
    return _file_function("guess_mime_type", value)


def file_same_location(
    left: vane.File | vane.Expression,
    right: vane.File | vane.Expression,
) -> vane.Expression:
    return _file_function("file_same_location", left, right)


def file_same_content(
    left: vane.File | vane.Expression,
    right: vane.File | vane.Expression,
) -> vane.Expression:
    return _file_function("file_same_content", left, right)


def file_locator_id(value: vane.File | vane.Expression) -> vane.Expression:
    return _file_function("file_locator_id", value)


def file_content_id(value: vane.File | vane.Expression) -> vane.Expression:
    return _file_function("file_content_id", value)


def list_files(
    path: str,
    recursive: bool = False,
    *,
    connection: vane.DuckDBPyConnection | None = None,
) -> vane.DuckDBPyRelation:
    """Return deterministic metadata rows from the SQL ``list_files`` function."""
    return vane.table_function("list_files", [path, recursive], connection=connection)


def from_files(
    path: str | list[str],
    *,
    connection: vane.DuckDBPyConnection | None = None,
) -> vane.DuckDBPyRelation:
    """Return a one-column relation of canonical FILE values."""
    parameter: object = path
    if isinstance(path, list):
        parameter = vane.Value(path, vane.list_type(vane.sqltypes.VARCHAR))
    parameters = [parameter]
    return vane.table_function("list_files", parameters, connection=connection).select(vane.ColumnExpression("file"))


__all__ = [
    "VaneFileReader",
    "audio_file",
    "file",
    "file_content_id",
    "file_enrich",
    "file_exists",
    "file_locator_id",
    "file_mime_type",
    "file_path",
    "file_same_content",
    "file_same_location",
    "file_size",
    "file_stat",
    "from_files",
    "guess_mime_type",
    "image_file",
    "list_files",
    "open_file",
    "to_file",
    "try_to_file",
    "video_file",
]
