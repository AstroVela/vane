# Native video seeking measurements

`scripts/benchmark_video_seek.py` compares the retained Python backend, native
sequential decoding, and native indexed decoding of the same file. It measures
exact frame access, a late time window, sampled keyframes, and streaming output.
Each operation has one warmup; measured repetitions alternate execution order.
Index construction is reported separately, including its complete hashing and
decode passes. Query timings include binding, selection and pixel conversion.

The separate native diagnostic repeats the same selection cursor and reports
FILE content bytes read, returned decoded frames (including discarded preroll),
seeks and selected frames. It excludes output pixel conversion and codec-internal
lookahead. `--http` serves the same local file through a loopback HTTP range server
and additionally counts actual response-body bytes for timed queries. This is not
a cloud-storage benchmark. Index reads verify logical blocks and may read more
bytes than an unverified tiny selection.

Run from an installed environment outside the checkout, with a matching built
video artifact. The ordinary extension installation/loading and build workflow
is documented in [NATIVE_MEDIA_EXTENSIONS.md](../../NATIVE_MEDIA_EXTENSIONS.md).

```bash
export VANE_BENCH_REPO="$PWD"
cd /tmp
"$VANE_BENCH_REPO/.venv/bin/python" \
  "$VANE_BENCH_REPO/scripts/benchmark_video_seek.py" video-seek.mp4 \
  --extension "$VANE_BENCH_REPO/build/python-release/vane_extensions/video.duckdb_extension" \
  --start-time 60 --end-time 61 --idx 1500 --interval 0.5 \
  --height 90 --width 160 --threads 1 --repetitions 5 \
  --allow-unsigned-development-artifact
```

Repeat with `--http` for network byte accounting. Use `--mode python`,
`--mode sequential`, and `--mode indexed` in separate processes when comparing
process peak RSS. The peak includes imports, extension loading, index construction
and warmups, and cannot be reset between repetitions. Cache state is explicitly
uncontrolled at the operating-system level; query results after warming do not
establish cold-cache performance. The JSON output includes the engine identity,
codec/package versions, input/artifact hashes, dimensions, timing samples,
selection metrics and output sizes. Compare selected frame counts and output
sizes before attributing differences to seeking. Report the source revision and
CPU model alongside the captured output.

This deterministic synthetic fixture has 1,800 frames, a 24-frame GOP, two B
frames, a constant 24 FPS rate and changing image content. Generate it with the
same installed PyAV/NumPy versions recorded for the measurements:

```python
import av
import numpy as np

rng = np.random.default_rng(714)
background = rng.integers(0, 256, (240, 320, 3), dtype=np.uint8)
with av.open("video-seek.mp4", "w") as output:
    stream = output.add_stream("mpeg4", rate=24)
    stream.width, stream.height, stream.pix_fmt = 320, 240, "yuv420p"
    stream.codec_context.gop_size = 24
    stream.codec_context.max_b_frames = 2
    for index in range(1800):
        pixels = np.roll(background, index % 320, axis=1)
        frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
        for packet in stream.encode(frame):
            output.mux(packet)
    for packet in stream.encode():
        output.mux(packet)
```

Index construction amortizes across repeated requests. A one-off request includes
that initial full-file work if no index exists. Codecs with little decode cost,
short clips, dense selections or large index-to-content ratios can regress;
measure those cases before selecting indexed execution for an application.
