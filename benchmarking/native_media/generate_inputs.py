#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0
"""Generate deterministic, synthetic inputs for benchmark_native_media.py."""

from __future__ import annotations

import argparse
import hashlib
import json
import wave
from pathlib import Path

import av
import numpy as np
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(757)
    paths = []
    for label, height, width in [("small", 32, 32), ("large", 720, 1280)]:
        path = args.output / f"image-{label}.png"
        pixels = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
        Image.fromarray(pixels).save(path)
        paths.append(path)
    for label, rate, seconds in [("small", 8000, 0.1), ("large", 48000, 5)]:
        path = args.output / f"audio-{label}.wav"
        t = np.arange(int(rate * seconds)) / rate
        samples = (np.column_stack([np.sin(2 * np.pi * 440 * t), np.sin(2 * np.pi * 660 * t)]) * 12000).astype("<i2")
        with wave.open(str(path), "wb") as output:
            output.setparams((2, 2, rate, len(samples), "NONE", "uncompressed"))
            output.writeframes(samples.tobytes())
        paths.append(path)
    for label, height, width, frames in [("small", 24, 32, 12), ("large", 720, 1280, 60)]:
        path = args.output / f"video-{label}.mp4"
        with av.open(str(path), "w") as output:
            stream = output.add_stream("mpeg4", rate=30)
            stream.height, stream.width, stream.pix_fmt = height, width, "yuv420p"
            stream.codec_context.gop_size = 15
            stream.codec_context.max_b_frames = 2
            y, x = np.indices((height, width))
            for index in range(frames):
                pixels = np.stack([(x + index * 7) % 256, (y + index * 3) % 256, (x + y + index * 5) % 256], axis=-1)
                frame = av.VideoFrame.from_ndarray(pixels.astype(np.uint8), format="rgb24")
                for packet in stream.encode(frame):
                    output.mux(packet)
            for packet in stream.encode():
                output.mux(packet)
        paths.append(path)
    print(
        json.dumps(
            {
                path.name: {"bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                for path in paths
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
