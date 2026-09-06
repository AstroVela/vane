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
    parser.add_argument(
        "--matrix", action="store_true", help="include JPEG, additional audio formats, rates and channel counts"
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(757)
    paths = []
    for label, height, width in [("small", 32, 32), ("large", 720, 1280)]:
        path = args.output / f"image-{label}.png"
        pixels = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
        Image.fromarray(pixels).save(path)
        paths.append(path)
        if args.matrix:
            for mode in ("RGB", "L"):
                jpeg = args.output / f"image-{label}-{mode}.jpg"
                Image.fromarray(pixels).convert(mode).save(jpeg, quality=90)
                paths.append(jpeg)
    for label, rate, seconds in [("small", 8000, 0.1), ("large", 48000, 5)]:
        path = args.output / f"audio-{label}.wav"
        t = np.arange(int(rate * seconds)) / rate
        samples = (np.column_stack([np.sin(2 * np.pi * 440 * t), np.sin(2 * np.pi * 660 * t)]) * 12000).astype("<i2")
        with wave.open(str(path), "wb") as output:
            output.setparams((2, 2, rate, len(samples), "NONE", "uncompressed"))
            output.writeframes(samples.tobytes())
        paths.append(path)
    if args.matrix:
        import soundfile

        audio_specs = [
            ("small", 8000, 0.1, 2),
            ("large", 48000, 5, 2),
            ("long", 48000, 30, 2),
            ("mono", 44100, 5, 1),
            ("surround", 48000, 5, 6),
            ("high-rate", 96000, 1, 8),
        ]
        for label, rate, seconds, channels in audio_specs:
            times = np.arange(int(rate * seconds)) / rate
            samples = (
                np.stack([np.sin(2 * np.pi * (440 + 220 * channel) * times) for channel in range(channels)], axis=1)
                * 12000
            ).astype("int16")
            for format in ("WAV", "FLAC", "AIFF"):
                path = args.output / f"audio-{label}.{format.lower()}"
                if path not in paths:
                    soundfile.write(path, samples, rate, format=format, subtype="PCM_16")
                    paths.append(path)
            if channels == 2 and label in ("small", "large"):
                for format, subtype in (("MP3", "MPEG_LAYER_III"), ("OGG", "VORBIS")):
                    path = args.output / f"audio-{label}.{format.lower()}"
                    soundfile.write(path, samples, rate, format=format, subtype=subtype)
                    paths.append(path)
                # AAC/Opus encode at 48 kHz; codec delay/padding is part of each
                # recorded input and may yield different backend frame counts.
                rate = 48000
                times = np.arange(int(rate * seconds)) / rate
                encoded_samples = np.stack(
                    [np.sin(2 * np.pi * (440 + 220 * channel) * times) for channel in range(2)]
                ).astype("float32") * (12000 / 32768)
                for container, codec in (("adts", "aac"), ("mp4", "aac"), ("webm", "libopus")):
                    path = args.output / f"audio-{label}.{container}"
                    with av.open(str(path), "w", format=container) as output:
                        stream = output.add_stream(codec, rate=rate)
                        stream.layout = "stereo"
                        stream.bit_rate = 96000
                        frame = av.AudioFrame.from_ndarray(encoded_samples, format="fltp", layout="stereo")
                        frame.sample_rate = rate
                        for packet in stream.encode(frame):
                            output.mux(packet)
                        for packet in stream.encode():
                            output.mux(packet)
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
