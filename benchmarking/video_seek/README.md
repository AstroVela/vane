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
process peak RSS. Python-only mode does not load the native extension;
sequential-only mode does not build an unused index. The peak includes each
mode's imports, extension loading and index construction where applicable,
and warmups, and cannot be reset between repetitions. Cache state is explicitly
uncontrolled at the operating-system level; query results after warming do not
establish cold-cache performance. The JSON output includes the engine identity,
codec/package versions, input/artifact hashes, dimensions, timing samples,
selection metrics and output sizes. Compare selected frame counts and output
sizes before attributing differences to seeking. Report the source revision and
CPU model alongside the captured output.

This deterministic synthetic fixture has 1,800 frames, a 24-frame GOP cap, two B
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

## Measurements from 2026-09-06

Native implementation: `eabaf1a6fb44df7abf5c3e0315563074844778e6`; engine
`v1.5.5-vane.eabaf1a6fb`, SourceID `d0db5a3a29`, Vane `0.2.0.dev625`.
Release build with GCC 13.3.0, native FFmpeg 8.1.1, Python 3.12.3, PyAV 17.1.0
and NumPy 2.5.2. The shared Linux x86-64 host has an Intel Xeon E5-2686 v4
2.30 GHz CPU (18 cores / 36 logical CPUs), about 62 GiB RAM and glibc 2.39.
Queries use the local-fast runner with one DuckDB thread and one codec thread.
Each row reports the median of five repetitions after one warmup. OS cache
state is uncontrolled. No encoded-video temporary-file materialization occurs.

The fixture above is 3,379,065 bytes and contains 1,800 decoded frames and 76
keyframes. Its SHA-256 is
`32084b1e0563397ac85241e00d38e3ebd03565caa39a3828155762380d8f8413`.
The tested unsigned development video artifact has SHA-256
`51a27c0c51657f92f383c86d886a98c22d148f0d11bd7ffc28f886b13b6cd7d5`.

Index construction takes 1.669 s locally (1.663 s CPU) and 1.831 s over loopback
HTTP (1.863 s CPU including the server). Both read 6,860,395 FILE bytes and
produce a 160,248-byte index. These costs are excluded from the reused-index
query timings below. For repeated exact-frame queries on this local fixture,
about eleven requests amortize construction relative to native sequential reads.
The benchmark holds the index BLOB in memory between requests; persistent-store
index retrieval is outside these timings.

Exact access selects frame 1,500 at its original 320 × 240 RGB dimensions.
The other operations select the inclusive 60–61 s window, sample every 0.5 s
and resize to 160 × 90 RGB. Window and stream return three frames; keyframes
return two. All modes return matching frame counts and payload sizes for
every measured repetition. Cross-backend pixel equality is not asserted.

| Storage / operation | Python ms | Native sequential ms | Native indexed ms |
| --- | ---: | ---: | ---: |
| Local / Exact frame | 283.99 | 172.47 | 16.24 |
| Local / Window list | 345.45 | 206.89 | 17.35 |
| Local / Keyframe list | 345.16 | 206.28 | 8.05 |
| Local / Streaming source | 345.07 | 215.44 | 23.42 |
| HTTP / Exact frame | 309.71 | 228.97 | 23.97 |
| HTTP / Window list | 370.62 | 275.03 | 25.62 |
| HTTP / Keyframe list | 378.05 | 273.64 | 16.31 |
| HTTP / Streaming source | 376.52 | 276.36 | 30.69 |

The same native cursor diagnostics report these counts for both storage modes:

| Operation | Sequential → indexed decoded frames | Sequential → indexed FILE bytes | Indexed seeks |
| --- | ---: | ---: | ---: |
| Exact frame | 1,501 → 13 | 2,967,130 → 364,409 | 1 |
| Window list | 1,800 → 14 | 3,462,567 → 429,945 | 2 |
| Keyframe list | 1,800 → 2 | 3,462,567 → 429,945 | 2 |
| Streaming source | 1,800 → 14 | 3,462,567 → 429,945 | 2 |

HTTP response-body accounting agrees with the native FILE-byte counters on every
measured repetition. The Python backend transfers 4,212,314 bytes for exact
access and 4,445,607 bytes for the other late-window operations. Native sequential
and indexed reads use the same pinned codec; their difference includes seeking
and index verification. Python has its own codec, buffering and conversion path.

Local exact-access median CPU time is 283.95 / 172.46 / 16.25 ms for Python /
native sequential / indexed. HTTP values are 311.11 / 243.58 / 26.01 ms, including
the loopback server in process CPU accounting. Each JSON report also emits all
per-operation wall/CPU samples and output sizes.

Separate processes for the same late-window workload report these peak RSS values:

| Mode | Peak RSS MiB | Included index construction |
| --- | ---: | --- |
| python | 234.00 | No |
| sequential | 216.58 | No |
| indexed | 224.90 | Yes |

These are whole-process high-water marks, including startup and warmups, rather
than an operator's working-memory measurement. Python-only mode does not load
the native video artifact; the two native modes do.

### Cases where indexing adds cost

The dense workload reuses the large fixture with `--start-time 0 --end-time 75`,
`--interval 0.041666666666666664 --height 48 --width 64`. All modes return 1,800
frames for the list and stream, and 76 keyframes. Native sequential and indexed
list latency is 556.80 and 1,707.45 ms; streaming is 568.36 and 1,717.40 ms.
Both decode all 1,800 frames for those two operations. Index verification adds
work without skipping decoding, making indexed execution about three times
slower. Indexed FILE reads are 3,481,330 bytes versus 3,462,567 sequentially.

A smaller fixture contains 240 random-noise 96 × 64 frames at 24 FPS, with a
12-frame GOP cap, two B frames and the encoder default bitrate. Generate it
with the same package versions:

```python
from fractions import Fraction
import av
import numpy as np

rng = np.random.default_rng(714)
with av.open("small-video-seek.mp4", "w", format="mp4") as output:
    stream = output.add_stream("mpeg4", rate=24)
    stream.width, stream.height, stream.pix_fmt = 96, 64, "yuv420p"
    stream.codec_context.gop_size = 12
    stream.codec_context.max_b_frames = 2
    stream.codec_context.time_base = Fraction(1, 24)
    for index in range(240):
        pixels = rng.integers(0, 256, (64, 96, 3), dtype=np.uint8)
        frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
        frame.pts, frame.time_base = index, Fraction(1, 24)
        for packet in stream.encode(frame):
            output.mux(packet)
    for packet in stream.encode():
        output.mux(packet)
```

Its size is 216,776 bytes and SHA-256 is
`1c8cf8b3dc51b53a53ea41e457a4710db8e37eb612d14d7f564e433f805860d4`.
The command uses `--idx 200 --start-time 8 --end-time 9 --interval 0.5
--height 32 --width 48`, with the same thread/warmup/repetition settings.

Exact access falls from 7.18 to 3.48 ms natively and decodes 201 versus three
frames, but verified-block and container-probe overhead increases FILE bytes
from 265,566 to 302,480. Its 21,432-byte index costs 288.77 ms to construct:
roughly 78 repeated exact-access requests amortize that cost. An isolated
request which first builds the index is more expensive than a sequential read.
