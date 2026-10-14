# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Typed Arrow IPC for selected, time-aligned content between video stages."""

from __future__ import annotations

from typing import Any

import numpy as np
import pyarrow as pa

from vane.ai._audio_embedding import AudioClip
from vane.ai._video_embedding import VideoClip
from vane.ai._video_pipeline_types import TEXT_MODALITIES, TimedContent

_FRAME = pa.struct(
    [
        ("index", pa.int64()),
        ("time", pa.float64()),
        ("height", pa.int32()),
        ("width", pa.int32()),
        ("rgb", pa.binary()),
    ]
)
_CONTENT = pa.schema(
    [
        ("start", pa.float64()),
        ("end", pa.float64()),
        ("text", pa.string()),
        ("frames", pa.list_(_FRAME)),
        ("samples", pa.binary()),
        ("sample_rate", pa.int32()),
        ("channels", pa.int32()),
    ]
)


def encode_content(channel: str, values: tuple[TimedContent, ...]) -> bytes:
    rows = []
    for item in values:
        row: dict[str, Any] = {"start": item.start, "end": item.end}
        if channel in TEXT_MODALITIES:
            row["text"] = item.value
        elif channel == "visual":
            row["frames"] = [
                {"index": i, "time": t, "height": frame.shape[0], "width": frame.shape[1], "rgb": frame.tobytes()}
                for i, t, frame in zip(item.value.frame_indices, item.value.frame_times, item.value.frames, strict=True)
            ]
        else:
            row.update(
                samples=np.asarray(item.value.samples, dtype="<f8").tobytes(),
                sample_rate=item.value.sample_rate,
                channels=item.value.samples.shape[1],
            )
        rows.append(row)
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, _CONTENT) as writer:
        writer.write_table(pa.Table.from_pylist(rows, schema=_CONTENT))
    return sink.getvalue().to_pybytes()


def decode_content(channel: str, data: bytes) -> tuple[TimedContent, ...]:
    table = pa.ipc.open_stream(data).read_all()
    if table.schema != _CONTENT:
        raise ValueError("invalid video content IPC schema")
    result = []
    for row in table.to_pylist():
        if channel in TEXT_MODALITIES:
            value = row["text"]
        elif channel == "visual":
            frames = row["frames"]
            value = VideoClip(
                frames=tuple(
                    np.frombuffer(f["rgb"], dtype=np.uint8).reshape(f["height"], f["width"], 3) for f in frames
                ),
                frame_indices=tuple(f["index"] for f in frames),
                frame_times=tuple(f["time"] for f in frames),
            )
        else:
            value = AudioClip(
                samples=np.frombuffer(row["samples"], dtype="<f8").reshape(-1, row["channels"]),
                sample_rate=row["sample_rate"],
            )
        result.append(TimedContent(row["start"], row["end"], value))
    return tuple(result)
