# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Public streaming video query construction; backend choice belongs to the binder."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import vane
    from vane.datasource.video_reader import _ImageVideoFrameSource


def read_video_frames(
    path: str | os.PathLike[str] | vane.File | list[str | os.PathLike[str] | vane.File] | None,
    image_height: int,
    image_width: int,
    is_key_frame: bool | None = None,
    *,
    sample_interval_seconds: int | float | None = None,
    start_time: int | float = 0,
    end_time: int | float | None = None,
    max_input_bytes: int = 8 * 1024**3,
    max_decoded_frames: int = 1_000_000,
    max_pixels: int = 32 * 1024**2,
    max_partition_bytes: int = 10 * 1024**2,
    frame_limit: int | None = None,
    read_task_count: int | None = None,
    on_error: str = "raise",
    indexes: list[bytes] | None = None,
    connection: vane.DuckDBPyConnection | None = None,
) -> vane.DuckDBPyRelation:
    """Stream selected frames with ``data: IMAGE('RGB', image_height, image_width)``.

    Inputs are exact paths or FILE views, without directory/glob discovery.
    NULL/empty input produces no rows. Construction and binding do not open files.
    Each row retains ``path``, ``file`` and presentation-order frame metadata.

    ``video_backend`` chooses Python or the loaded native video extension when
    binding. Time windows include both endpoints. ``on_error='skip'`` skips only
    encoded format failures; I/O, permissions, resource limits and cancellation
    propagate. Row order across file tasks is unspecified; use ORDER BY if needed.
    ``indexes`` explicitly selects native keyframe seeking, with one index BLOB
    from ``build_video_index`` per input FILE. Sources are opened on each Worker.
    """
    import vane

    if path is None:
        inputs = []
    elif isinstance(path, (str, os.PathLike, vane.File)):
        inputs = [path]
    elif isinstance(path, list):
        inputs = path
    else:
        raise TypeError("path must be a path, FILE value, list of paths/FILE values, or None")
    files = []
    for value in inputs:
        if isinstance(value, vane.VideoFile):
            files.append(value)
        elif type(value) is vane.File:
            files.append(vane.VideoFile(value.url, value.content_type, value.position, value.size, value.checksum))
        elif isinstance(value, (str, os.PathLike)):
            url = os.fspath(value)
            if not isinstance(url, str):
                raise TypeError("video paths must be strings")
            files.append(vane.VideoFile(url))
        else:
            raise TypeError("video path lists require paths or FILE/VIDEOFILE values; NULL elements are invalid")

    if indexes is not None:
        if not isinstance(indexes, list) or any(not isinstance(index, bytes) for index in indexes):
            raise TypeError("indexes must be a list of bytes or None")
        if len(indexes) != len(files):
            raise ValueError("indexes must correspond to the FILE views")

    options: dict[str, object] = {
        "start_time": start_time,
        "end_time": end_time,
        "is_key_frame": is_key_frame,
        "sample_interval_seconds": sample_interval_seconds,
        "max_input_bytes": max_input_bytes,
        "max_decoded_frames": max_decoded_frames,
        "max_pixels": max_pixels,
        "max_partition_bytes": max_partition_bytes,
        "frame_limit": frame_limit,
        "on_error": on_error,
        "read_task_count": read_task_count,
        "indexes": indexes,
    }
    integers: dict[str, object] = {"image_height": image_height, "image_width": image_width}
    integers.update(
        {name: options[name] for name in ("max_input_bytes", "max_decoded_frames", "max_pixels", "max_partition_bytes")}
    )
    for name in ("frame_limit", "read_task_count"):
        if options[name] is not None:
            integers[name] = options[name]
    for name, integer_value in integers.items():
        if isinstance(integer_value, bool) or not isinstance(integer_value, int):
            raise TypeError(f"{name} must be int")
    for name in ("start_time", "end_time", "sample_interval_seconds"):
        time_value = options[name]
        if time_value is not None and (isinstance(time_value, bool) or not isinstance(time_value, (int, float))):
            raise TypeError(f"{name} must be int, float, or None")
    if is_key_frame is not None and not isinstance(is_key_frame, bool):
        raise TypeError("is_key_frame must be bool or None")
    parameters = [
        vane.Value(files, vane.list_type(vane.file_type(vane.MediaType.video()))),
        image_height,
        image_width,
    ]
    con = vane.default_connection() if connection is None else connection
    return con._read_video_frames(parameters, options)


def _image_video_source(
    files: list[vane.VideoFile],
    height: int,
    width: int,
    start_time: float,
    end_time: float | None,
    is_key_frame: bool | None,
    sample_interval_seconds: float | None,
    max_input_bytes: int,
    max_decoded_frames: int,
    max_pixels: int,
    max_partition_bytes: int,
    frame_limit: int | None,
    on_error: str,
    read_task_count: int | None,
) -> _ImageVideoFrameSource:
    from vane.datasource.video_reader import _ImageVideoFrameSource

    return _ImageVideoFrameSource(
        files,
        height=height,
        width=width,
        start_time=start_time,
        end_time=end_time,
        is_key_frame=is_key_frame,
        sample_interval_seconds=sample_interval_seconds,
        max_input_bytes=max_input_bytes,
        max_decoded_frames=max_decoded_frames,
        max_pixels=max_pixels,
        max_partition_bytes=max_partition_bytes,
        frame_limit=frame_limit,
        on_error=on_error,
        read_task_count=read_task_count,
    )


__all__ = ["read_video_frames"]
