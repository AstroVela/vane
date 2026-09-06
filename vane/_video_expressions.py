# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Bounded video frame expressions; codec work starts only during execution."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import vane
from vane._expressions import as_expression
from vane._video_file import (
    DEFAULT_VIDEO_BUFFER_SIZE,
    DEFAULT_VIDEO_MAX_FRAMES,
    DEFAULT_VIDEO_MAX_INPUT_BYTES,
    DEFAULT_VIDEO_MAX_PIXELS,
    _close_image,
    _iter_video_frames,
    _load_av,
    _load_pillow,
    _normalize_frame_options,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Generator

    from vane._native import _DataSourceExecutionContext

DEFAULT_VIDEO_OUTPUT_BYTES = 64 * 1024**2
DEFAULT_VIDEO_OUTPUT_FRAMES = 10_000


def _argument(
    value: Any, name: str, kind: type, maximum: int | None = None, *, optional: bool = False
) -> vane.Expression:
    if isinstance(value, vane.Expression):
        return value
    if optional and value is None:
        return vane.ConstantExpression(None)
    accepted = (int, float) if kind is float else (kind,)
    if not isinstance(value, accepted) or (kind is not bool and isinstance(value, bool)):
        raise TypeError(f"{name} must be {kind.__name__}{' or None' if optional else ''}, or Expression")
    if kind is float:
        assert isinstance(value, (int, float))
        if not math.isfinite(value) or value < 0 or (name == "sample_interval_seconds" and value == 0):
            raise ValueError(
                f"{name} must be finite and {'positive' if name == 'sample_interval_seconds' else 'nonnegative'}"
            )
    elif maximum is not None:
        assert isinstance(value, int)
        if value <= 0 or value > maximum:
            raise ValueError(f"{name} must be between 1 and {maximum}")
    elif name == "idx":
        assert isinstance(value, int)
        if value < 0 or value >= 2**63:
            raise ValueError("idx must be a nonnegative BIGINT")
    elif name == "on_error" and value not in ("raise", "null"):
        raise ValueError("on_error must be 'raise' or 'null'")
    return vane.ConstantExpression(value)


def _frame_expression(name: str, value: vane.VideoFile | vane.Expression, options: dict[str, Any]) -> vane.Expression:
    arguments = [as_expression(value)]
    for key, option in options.items():
        if key in ("start_time", "end_time", "sample_interval_seconds"):
            expression = _argument(option, key, float, optional=key != "start_time")
        elif key in ("width", "height"):
            expression = _argument(option, key, int, 100_000, optional=True)
        elif key == "is_key_frame":
            expression = _argument(option, key, bool, optional=True)
        elif key == "on_error":
            expression = _argument(option, key, str)
        elif key == "idx":
            expression = _argument(option, key, int, optional=name == "video_scan_stats")
        elif key == "index":
            expression = _argument(option, key, bytes, optional=True)
        else:
            maximum = {
                "max_input_bytes": 16 * 1024**3,
                "max_decoded_frames": 100_000_000,
                "max_pixels": DEFAULT_VIDEO_MAX_PIXELS,
                "max_output_bytes": 256 * 1024**2,
                "max_output_frames": 100_000,
                "max_index_bytes": 64 * 1024**2,
            }[key]
            expression = _argument(option, key, int, maximum)
        arguments.append(expression)
    return vane.FunctionExpression(name, *arguments)


def video_frames(
    value: vane.VideoFile | vane.Expression,
    *,
    start_time: int | float | vane.Expression = 0,
    end_time: int | float | vane.Expression | None = None,
    width: int | vane.Expression | None = None,
    height: int | vane.Expression | None = None,
    is_key_frame: bool | vane.Expression | None = None,
    sample_interval_seconds: int | float | vane.Expression | None = None,
    on_error: str | vane.Expression = "raise",
    max_input_bytes: int | vane.Expression = DEFAULT_VIDEO_MAX_INPUT_BYTES,
    max_decoded_frames: int | vane.Expression = DEFAULT_VIDEO_MAX_FRAMES,
    max_pixels: int | vane.Expression = DEFAULT_VIDEO_MAX_PIXELS,
    max_output_bytes: int | vane.Expression = DEFAULT_VIDEO_OUTPUT_BYTES,
    max_output_frames: int | vane.Expression = DEFAULT_VIDEO_OUTPUT_FRAMES,
    index: bytes | vane.Expression | None = None,
) -> vane.Expression:
    """Return a bounded list of RGB Image frame records with VIDEOFILE provenance."""
    options = locals().copy()
    del options["value"]
    return _frame_expression("video_frames", value, options)


def video_keyframes(
    value: vane.VideoFile | vane.Expression,
    *,
    start_time: int | float | vane.Expression = 0,
    end_time: int | float | vane.Expression | None = None,
    width: int | vane.Expression | None = None,
    height: int | vane.Expression | None = None,
    sample_interval_seconds: int | float | vane.Expression | None = None,
    on_error: str | vane.Expression = "raise",
    max_input_bytes: int | vane.Expression = DEFAULT_VIDEO_MAX_INPUT_BYTES,
    max_decoded_frames: int | vane.Expression = DEFAULT_VIDEO_MAX_FRAMES,
    max_pixels: int | vane.Expression = DEFAULT_VIDEO_MAX_PIXELS,
    max_output_bytes: int | vane.Expression = DEFAULT_VIDEO_OUTPUT_BYTES,
    max_output_frames: int | vane.Expression = DEFAULT_VIDEO_OUTPUT_FRAMES,
    index: bytes | vane.Expression | None = None,
) -> vane.Expression:
    """Return a bounded list of keyframes as native RGB Image values."""
    options = locals().copy()
    del options["value"]
    return _frame_expression("video_keyframes", value, options)


def get_video_frame_by_idx(
    value: vane.VideoFile | vane.Expression,
    idx: int | vane.Expression,
    *,
    on_error: str | vane.Expression = "raise",
    max_input_bytes: int | vane.Expression = DEFAULT_VIDEO_MAX_INPUT_BYTES,
    max_decoded_frames: int | vane.Expression = DEFAULT_VIDEO_MAX_FRAMES,
    max_pixels: int | vane.Expression = DEFAULT_VIDEO_MAX_PIXELS,
    max_output_bytes: int | vane.Expression = DEFAULT_VIDEO_OUTPUT_BYTES,
    index: bytes | vane.Expression | None = None,
) -> vane.Expression:
    """Return one native Image at a zero-based presentation-order frame index."""
    options = locals().copy()
    del options["value"]
    return _frame_expression("get_video_frame_by_idx", value, options)


def build_video_index(
    value: vane.VideoFile | vane.Expression,
    *,
    max_input_bytes: int | vane.Expression = DEFAULT_VIDEO_MAX_INPUT_BYTES,
    max_decoded_frames: int | vane.Expression = DEFAULT_VIDEO_MAX_FRAMES,
    max_pixels: int | vane.Expression = DEFAULT_VIDEO_MAX_PIXELS,
    max_index_bytes: int | vane.Expression = 64 * 1024**2,
) -> vane.Expression:
    """Build a bounded reusable video index with the selected native backend.

    Execution reads and decodes the source once, plus a content verification pass.
    The returned BLOB binds the FILE view, codec build and decoded frame order.
    """
    options = locals().copy()
    del options["value"]
    return _frame_expression("build_video_index", value, options)


def video_index_info(index: bytes | vane.Expression) -> vane.Expression:
    """Inspect a video index's frame counts, size, and creation I/O without FILE I/O."""
    return vane.FunctionExpression("video_index_info", _argument(index, "index", bytes))


def video_scan_stats(
    value: vane.VideoFile | vane.Expression,
    *,
    start_time: int | float | vane.Expression = 0,
    end_time: int | float | vane.Expression | None = None,
    is_key_frame: bool | vane.Expression | None = None,
    sample_interval_seconds: int | float | vane.Expression | None = None,
    idx: int | vane.Expression | None = None,
    index: bytes | vane.Expression | None = None,
    max_input_bytes: int | vane.Expression = DEFAULT_VIDEO_MAX_INPUT_BYTES,
    max_decoded_frames: int | vane.Expression = DEFAULT_VIDEO_MAX_FRAMES,
    max_pixels: int | vane.Expression = DEFAULT_VIDEO_MAX_PIXELS,
) -> vane.Expression:
    """Execute native selection and report reads, decoded frames and seeks.

    This diagnostic repeats selection without converting or returning image pixels.
    NULL index selects sequential decoding; a supplied index selects verified seeks.
    """
    options = dict(
        start_time=start_time,
        end_time=end_time,
        is_key_frame=is_key_frame,
        sample_interval_seconds=sample_interval_seconds,
        max_input_bytes=max_input_bytes,
        max_decoded_frames=max_decoded_frames,
        max_pixels=max_pixels,
        idx=idx,
        index=index,
    )
    return _frame_expression("video_scan_stats", value, options)


def _scalar_video_frames(
    value: vane.VideoFile,
    options: dict[str, Any],
    execution_context: _DataSourceExecutionContext,
    reserve: Callable[[int, int], None],
) -> Generator[tuple[Any, ...], None, None]:
    """One frame at a time across the Python bridge; no connection lookup."""
    normalized = _normalize_frame_options(buffer_size=DEFAULT_VIDEO_BUFFER_SIZE, **options)
    av_module = _load_av()
    image_module = _load_pillow()
    frames = _iter_video_frames(value, normalized, av_module, image_module, None, execution_context)
    try:
        for frame in frames:
            image = frame.data
            try:
                if image.mode != "RGB":
                    raise RuntimeError("video decoder returned a non-RGB frame")
                width, height = image.size
                # C++ reserves row and batch payload before tobytes allocates a
                # second pixel buffer. Format-error NULLs do not refund it.
                reserve(width, height)
                payload = image.tobytes()
                if len(payload) != width * height * 3:
                    raise RuntimeError("video decoder returned an invalid RGB payload")
                base = frame.frame_time_base
                yield (
                    frame.frame_index,
                    frame.frame_time,
                    None if base is None else base.numerator,
                    None if base is None else base.denominator,
                    frame.frame_pts,
                    frame.frame_dts,
                    frame.frame_duration,
                    frame.is_key_frame,
                    width,
                    height,
                    payload,
                )
                del payload
            finally:
                _close_image(image)
    except BaseException:
        # Cleanup must not turn a system/control-flow failure into a format
        # error that the SQL on_error='null' policy could suppress.
        try:
            frames.close()
        except BaseException:
            pass
        raise
    else:
        frames.close()


__all__ = [
    "build_video_index",
    "get_video_frame_by_idx",
    "video_frames",
    "video_index_info",
    "video_keyframes",
    "video_scan_stats",
]
