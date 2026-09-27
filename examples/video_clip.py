# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Write one bounded clip, without model calls.

Install vane-ai[video], then run:
    python examples/video_clip.py input.mp4 output.mp4 --start 10 --end 20

VideoFile.clip executes eagerly in this process. For distributed execution use
vane.video_clip(vane.col("video"), start, end) in a relation, or SQL:
    SELECT video_clip(file, 10, 20) AS clip FROM videos;

The result carries encoded bytes, MIME type, source start/end, duration, frame
count and audio presence. The caller owns persistence; Ray workers never
return worker-local temporary paths. Every worker must be able to read the
source FILE and must have vane-ai[video] installed.

Bounds are relative to the first non-attached video stream's declared origin.
The interval is half-open. The picture displayed at the start is retained for
the remaining part of its interval, even if its original PTS precedes start.
Add result.start_time to clip timestamps to map citations to the source.
Video timestamps use a 1/60000-second grid; audio uses its source sample rate.
Audio timestamps round to the nearest sample, with ties toward later time.
The exclusive end can be shortened by less than one video tick.

Output is MP4 with MPEG-4 Part 2 video (yuv420p) and the first audio stream as
AAC. YUV matrix/range is converted to BT.709 limited range; source color
primaries and transfer characteristics are preserved and must remain constant.
The MP4 movie clock represents both the video and audio sample grids exactly.
Silent input stays silent; include_audio=False explicitly drops audio.
This is the Python/PyAV backend; selecting the native backend for the SQL
operator fails explicitly. No codec fallback, resizing, downmixing, subtitle
copying or metadata copying is performed. Odd or changing dimensions, decoder
sample aspect ratio changes, missing/nonmonotonic timestamps, display rotation,
unsupported audio rates/layouts, and windows outside the video fail.
A ratio change encountered while decoding is rejected even if reordered
pictures precede that change on the presentation timeline.
A declared video frame rate is required.
Only mono/stereo audio is supported. Gaps in the selected track render as silence.

Defaults bound input to 1 GiB, output to 64 MiB, clip length to 300 s, decode
to 100000 video frames / 100 million audio samples per channel, visible pixels
to 32 Mi pixels, and execution to 60 s. SQL additionally bounds output per
chunk to 256 MiB. Decoding is sequential, so preceding frames count toward
limits. Timeout and cancellation checks surround I/O, codec calls and cleanup;
they do not preempt an atomic codec call, but an expired operation cannot
return a successful clip. Limit errors return no partial output.
"""

import argparse
from pathlib import Path

import vane


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input")
    parser.add_argument("output", type=Path)
    parser.add_argument("--start", required=True, type=float)
    parser.add_argument("--end", required=True, type=float)
    parser.add_argument("--without-audio", action="store_true")
    args = parser.parse_args()
    clip = vane.VideoFile(args.input).clip(args.start, args.end, include_audio=not args.without_audio)
    with args.output.open("xb") as output:
        output.write(clip.data)
    print(
        f"{clip.content_type}: source [{clip.start_time}, {clip.end_time}), {clip.frame_count} frames, audio={clip.has_audio}"
    )


if __name__ == "__main__":
    main()
