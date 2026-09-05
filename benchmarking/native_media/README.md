# Native extension measurements

These measurements compare explicitly selected Python and native backends at
`f7ee0fa475e21c2df48adbc2298a607714e5d349`, on 2026-09-05. They measure complete
local queries over repeated references to synthetic files. They do not isolate
interpreter overhead from codec algorithms, copying, or I/O behavior.

## Environment and method

- Intel Xeon E5-2686 v4, 18 physical cores / 36 logical CPUs, approximately
  62.7 GiB RAM; Linux 6.17.0, glibc 2.39; local NVMe/ext4 files.
- Release build, Vane 0.2.0.dev622, DuckDB `v1.5.5-vane.f7ee0fa475`, engine
  SourceID prefix `bf78c89787`. Each domain is loaded from its independently
  packaged, test-signed artifact before timing. The Python measurements also
  load the artifact, keeping connection setup consistent.
- Python 3.12.3, NumPy 2.5.2, Pillow 12.3.0, soundfile 0.14.0
  (libsndfile 1.2.2), soxr 1.1.0 (libsoxr 0.1.3-14-ga66f3ee), PyAV 17.1.0.
  Native extensions use pinned vcpkg FFmpeg 8.1.1#2. PyAV reports libavcodec
  62.28.101, libavformat 62.12.101, and libswscale 9.5.101.
- `local-fast` runner, one DuckDB thread unless specified; one warmup per
  backend, then five repetitions alternating backend order. Reported wall and
  process CPU times are medians. Input generation, imports, extension loading,
  and the input table setup are outside the timed region; query binding and
  execution are included. All runs use warm caches on a shared host without
  CPU pinning, so small differences should not be treated as significant.
- Peak RSS comes from additional fresh processes selecting only one backend.
  It includes imports, extension loading, warmups, and all five repetitions,
  and is sampled before hashing the files. It is not per-query allocation.
- PNG output is RGB at source resolution. WAV inputs are stereo PCM16;
  resampling requests interleaved Float64 at 16 kHz. Python uses SoXR HQ;
  native uses the pinned libswresample defaults. These quality settings are
  different, so the audio comparison does not establish equivalent fidelity.
- MPEG-4 inputs have 30 fps, GOP size 15, and up to two B-frames. Frame output
  is RGB Tensor at 160 by 90. Native resizing uses bilinear libswscale;
  Python uses PyAV's default reformatter. Frame counts and sums of global
  frame indices match for these inputs; pixel equality is not required.
- Image decode queries aggregate byte lengths; audio resampling queries
  aggregate frame counts; metadata queries aggregate widths or sample rates;
  video sources aggregate frame counts and indices. Results stay
  inside the engine until the aggregate is fetched. No decoded file export or
  network source is measured. Read-byte counts, temporary storage, remote
  latency, cancellation latency, and Ray scaling were not instrumented by
  this benchmark; those measurements remain part of #764.

## Reproduce

Use the installed environment and extension build procedure in
[NATIVE_MEDIA_EXTENSIONS.md](../../NATIVE_MEDIA_EXTENSIONS.md). Generate the
inputs outside the checkout; no generated media is committed:

```bash
python -I benchmarking/native_media/generate_inputs.py /tmp/vane-media-inputs
python -I scripts/benchmark_native_media.py image_decode \
  /tmp/vane-media-inputs/image-large.png \
  --extension build/python-release/vane_extensions/image.duckdb_extension \
  --rows 8 --repetitions 5 --threads 1 \
  --allow-unsigned-development-artifact
```

Repeat for each operation/input/row count below and its corresponding domain
artifact. Omit the unsigned-development flag when using a signed artifact
accepted by the runtime. The default `--backend both` alternates timings;
run `--backend python` and `--backend native` in separate processes for the
RSS comparison. Repeat the large-input cases with `--threads 4` for the
second table. The benchmark emits JSON containing all samples, medians,
aggregate results, versions, and input/artifact SHA-256 identities. Codec
version changes can change generated file hashes, particularly for MP4.

## Observations

Small-file workloads improve in these runs. Native PNG metadata reads a
smaller part of the file than the Python path, so its large-input gain cannot
be attributed solely to removing Python calls. Larger image decoding improves
by about 1.5 times. Larger audio resampling takes about 3.76 times as long
with the native backend, and native image/audio decoding has higher process
peak RSS here. These regressions are part of the result.

With two larger video sources, four configured threads reduce native wall
time from 239 ms to 152 ms; the Python path increases from 257 ms to 307 ms.
The scalar workloads have too few rows to establish parallel scaling and
show no consistent benefit from four threads. This is not a saturation study.

A separate late video window (`--start-time 1.5 --end-time 1.9`, two larger
files, one thread) returns 26 frames with an index sum of 1326 for each
backend: Python 171.94 ms, native 157.38 ms. Native scanning still decodes
from the beginning to retain exact global indices; this result does not
demonstrate indexed seek acceleration.

## Measurements

Wall and CPU columns show milliseconds per query; RSS shows MiB. The
ratio is Python wall time divided by native wall time (greater than one
means native was faster in that run).

| Operation | Input | Rows | Wall Python / native | Ratio | CPU Python / native | Peak RSS Python / native |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `image_metadata` | small | 1000 | 108.16 / 15.98 | 6.77 | 108.15 / 15.99 | 234.4 / 232.8 |
| `image_metadata` | large | 100 | 137.85 / 2.22 | 62.22 | 137.84 / 2.22 | 234.3 / 234.0 |
| `image_decode` | small | 100 | 31.61 / 18.42 | 1.72 | 31.62 / 18.43 | 232.6 / 247.2 |
| `image_decode` | large | 8 | 169.09 / 113.05 | 1.50 | 169.04 / 112.74 | 265.1 / 275.1 |
| `audio_metadata` | small | 100 | 17.74 / 12.29 | 1.44 | 17.74 / 12.30 | 229.3 / 240.0 |
| `audio_metadata` | large | 100 | 19.26 / 16.45 | 1.17 | 19.27 / 16.45 | 232.1 / 240.0 |
| `audio_resample` | small | 100 | 64.55 / 39.51 | 1.63 | 64.56 / 39.51 | 237.5 / 247.5 |
| `audio_resample` | large | 4 | 35.07 / 131.73 | 0.27 | 35.08 / 131.68 | 243.3 / 253.7 |
| `video_metadata` | small | 100 | 51.61 / 27.51 | 1.88 | 51.62 / 27.52 | 251.5 / 239.9 |
| `video_metadata` | large | 20 | 16.81 / 9.55 | 1.76 | 16.82 / 9.56 | 251.2 / 241.2 |
| `video_frames` | small | 10 | 62.16 / 21.05 | 2.95 | 62.15 / 21.06 | 266.9 / 251.7 |
| `video_frames` | large | 2 | 256.82 / 239.04 | 1.07 | 256.76 / 239.04 | 275.8 / 261.0 |

Four configured threads, same large inputs and row counts:

| Operation | Python wall ms | Native wall ms |
| --- | ---: | ---: |
| `image_metadata` | 125.82 | 2.18 |
| `image_decode` | 171.00 | 115.00 |
| `audio_metadata` | 19.89 | 16.71 |
| `audio_resample` | 35.23 | 131.91 |
| `video_metadata` | 16.13 | 9.43 |
| `video_frames` | 307.04 | 152.12 |

## Input identities

| File | Definition | Bytes | SHA-256 |
| --- | --- | ---: | --- |
| `image-small.png` | RGB 32 × 32 | 3172 | `15307ce943aad88fa626f001fdcd4396b81522d591791e1b9f5997ae0e48dec6` |
| `image-large.png` | RGB 1280 × 720 | 2769436 | `46051e478f1da432204a70b1ca6baa7f7e6f3afb1756ee8528faa0213ca53312` |
| `audio-small.wav` | 8 kHz stereo, 0.1 seconds | 3244 | `f8934a385ffebdaf4749c9a5e525ecf95483b0db94ee858b03efee37f78c5409` |
| `audio-large.wav` | 48 kHz stereo, 5 seconds | 960044 | `b3e8e261d3380129d82f96156c71fadce07c176d61686da6497e723372e046f5` |
| `video-small.mp4` | 32 × 24, 12 frames | 1337 | `e0672a2bc0c59b2d1915fadbb39a0c8d4b888ce21fbd41dbc225e2a220692060` |
| `video-large.mp4` | 1280 × 720, 60 frames | 476083 | `3498b9a16005dc4fc443bd6e7d7beb17b40f3d731e9fbe7701629f672b15f26b` |

Artifact SHA-256 identities used for these measurements (the test signature
is included in the digest):

- `image`: `10d9b52ddbd63ad767eba3414969ac6d6c173ae9dfe3a210c3993adcc0d703c2`
- `audio`: `3afe4f0626767b6f5629ee378ecb4f300d22b6d70e674b563d722cf7a95b47da`
- `video`: `474b62e07cc6ab5dec510f58b05dbdb3ac45e16f5d1c4347005cb76d9d629dfd`
