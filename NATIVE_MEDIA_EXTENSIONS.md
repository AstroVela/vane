# Native image, audio, and video extensions

Vane provides three optional DuckDB C++ extensions: `image`, `audio`, and
`video`. Each is a separate `.duckdb_extension` artifact and can be installed,
loaded, and selected independently. The base runtime continues to provide
FILE, its media subtypes, IMAGE, Tensor, FILE field access/comparison, and
governed I/O. Loading an extension does not change those types or enable its
backend automatically.

| Extension | Setting | Native operations |
| --- | --- | --- |
| `image` | `image_backend` | `image_file_metadata`, `decode_image_file` |
| `audio` | `audio_backend` | `audio_metadata`, `audio_resample` |
| `video` | `video_backend` | `video_metadata`, `VideoFrameSource` scanning |

IMAGE pixel operators belong to the image extension's domain; this change
implements the encoded-file operations listed above. The future public video
frame expressions and `read_video_frames` API remain tracked in #756.

## Select a backend

Install matching extension provider wheels using the existing optional-wheel
workflow in [DEVELOPMENT.md](DEVELOPMENT.md), then load the desired provider:

```python
import vane

con = vane.connect()
vane.load_installed_extension("image", connection=con)
con.execute("SET image_backend = 'native'")
con.sql("SELECT image_file_metadata(image_file('photo.png'))").show()
```

A locally installed DuckDB artifact can also be loaded with `LOAD image`.
Artifact names alone do not select a repository or download an unpublished
Vane extension. Distributed jobs use installed, trusted provider wheels as
described in [DISTRIBUTED_EXTENSIONS.md](DISTRIBUTED_EXTENSIONS.md).

All three settings default to `python` and accept only `python` or `native`.
They are also accepted by `vane.connect(config={"image_backend": "native"})`
and the equivalent configuration for the other domains.
A native request without its matching loaded extension fails while binding,
before FILE I/O. There is no automatic fallback. Set the corresponding option
back to `python` to select Python for newly bound queries. Python File value
methods such as `ImageFile.decode()` and `VideoFile.frames()` continue to use
their Python implementations; these SQL/connection settings govern SQL,
expressions, and the connection-bound video source.

The binder names native scalar functions explicitly in the plan. `EXPLAIN`
shows `native_image_file_metadata`, `native_decode_image_file`,
`native_audio_metadata`, `native_audio_resample`, or `native_video_metadata`.
Native video sources show `NATIVE_VIDEO_TENSOR_FRAMES`. Inspect the selected
setting with `current_setting('image_backend')`, and loaded artifacts with
`duckdb_extensions()`. Backend selection occurs when an expression is bound;
reusable prepared statements retain their bound implementation until rebound.
Lazy relations may be bound again when executed, and use the setting at that
binding. Set options before constructing and executing the query.

## Native contracts

The extensions call FFmpeg C libraries directly. Native media execution does
not import Pillow, soundfile, soxr, or PyAV. Python result conversion and an
explicitly registered Python filesystem remain separate boundaries. No
cross-backend bitwise/numerical compatibility is promised. MIME validation
uses container families: MP4/MOV and Matroska/WebM respectively share a
native demuxer and accepted MIME family.
Absent content types, `application/octet-stream`, and `binary/octet-stream`
allow format detection. A matching domain wildcard (`image/*`, `audio/*`,
or `video/*`) also permits the detected format; a different domain is rejected.

* Image supports encoded PNG and JPEG. Metadata reads headers without pixel
  decoding. Decode returns the native IMAGE type with 8-bit `L`, `LA`, `RGB`,
  or `RGBA` pixels. With no mode, palette and alpha-bearing decoded formats use RGBA,
  8-bit grayscale uses L, and other formats use RGB. PNG metadata may report
  P or I;16 even though decoded output uses these 8-bit modes. Unsupported
  formats and MIME mismatches follow `on_error='raise'|'null'`.
* Audio supports WAV, AIFF, FLAC, MP3, AAC, Ogg, MP4, and WebM containers with
  decoders in the pinned FFmpeg build. Metadata reports FFmpeg format/codec
  names. An exact frame count is returned only where PCM duration establishes
  it; otherwise it is NULL. Resampling uses libswresample and returns the
  existing STRUCT of interleaved Float64 samples, rate, frames, and channels.
  It uses the pinned libswresample defaults, while Python uses SoXR HQ;
  their quality settings and numerical outputs are different. Supported rates
  are 1..384000 Hz and channel counts are 1..64. This is
  distinct from the proposed Tensor-returning `resample` API in #755.
* Video supports MP4/MOV, Matroska/WebM, AVI, MPEG-TS, MPEG, and Ogg containers with
  decoders in the pinned build. Metadata preserves unknown values as NULL.
  The existing VideoFrameSource retains its RGB Tensor result schema. Its
  source association, zero-based frame index, PTS/DTS, duration, time base,
  and keyframe flag come from the selected stream and decoded frames.
  Times are relative to stream start when known, otherwise zero. Windows
  include both endpoints. Timestamp discontinuities reset sampling targets.
  Sampling tolerates a few DOUBLE rounding units at a threshold.

The video extension also registers the lower-level `native_video_frames`
table function, with the same positional scan parameters as
`native_video_tensor_frames`, but IMAGE output in its `frame` column. The
video extension can produce IMAGE without loading the image extension.
These are extension execution entry points; the public convenience API is
still VideoFrameSource.

Exact global frame indices currently require decoding from the beginning of
the stream, including for late time windows. This implementation does not
claim indexed seek acceleration; #714 owns that work. It does not materialize
non-seekable inputs to temporary files. Unsupported random access propagates
through the existing FILE reader.

## I/O and resource bounds

All codecs read through the executing query's ResolvedFile. Codec libraries
receive a logical byte stream, never the original URL. The existing FILE
resolver enforces position/size and selects the filesystem and Secret scope
on the executing Worker. Container-triggered external resource opens are
rejected. Native extensions add no credential fields or credential replay
mechanism.
The current resolver requires nonblocking opens and rejects registered Python
filesystems before their I/O callback. Native operators preserve this restriction,
including under `on_error='null'` or `'skip'`.

Metadata probes have byte budgets (image: 1 MiB default; audio/video: 8 MiB;
maximum: 64 MiB). FFmpeg probes additionally have a 30-second cooperative
deadline. Decoding checks input-view size, cumulative reads, dimensions,
decoded frames/samples, and output sizes. Cumulative codec reads are limited
to four times the configured input limit to account for probing and seeking.
Image/audio input limits may be set up to 4 GiB; video up to 16 GiB. The
hard pixel ceiling is 100 million, and decoded frame accounting allows at
most 512 MiB (audio decoder frames: 64 MiB). Decoder plane accounting includes alignment and is
conservative and can reject an image before its smaller converted output
would reach the output limit.

Image/audio output is bounded to 256 MiB per engine batch. Audio vector growth
may temporarily retain old and new buffers, up to 512 MiB in total. Video
emits bounded batches including FILE/provenance payload; source metadata is capped at 64 MiB and 100,000 FILE views. Frames from different
files can be decoded on separate threads or Workers. `read_task_count` selects
balanced groups of files for local tasks and Ray splits, capped at the number
of files; files within each group are processed sequentially. Its default
creates one group per file. A global frame limit uses one ordered work unit
regardless of `read_task_count`. Tensor output retains DuckDB's existing fixed
ARRAY vector reservations; the scan's payload budget is not a bound on those
engine reservations or total process RSS. Codec contexts, reference frames,
conversion buffers, and downstream query state also consume memory.

Cancellation is checked around I/O, packet/frame decoding, resampling, and
pixel conversion. Codec calls are cooperative boundaries, not preemptively
interruptible inside an individual codec call. `on_error` suppresses only
encoded-format failures. I/O, resource limits, allocation failures, and
cancellation propagate. Failed pixel allocations remain charged to the batch
budget even when their row is suppressed.

## Build and package

Base dependency/bootstrap and base wheel commands remain unchanged. Select
optional manifest features after bootstrapping the base dependencies. FFmpeg
also requires NASM on x86 (for example, the `nasm` package on Ubuntu):

```bash
"$VCPKG_ROOT/vcpkg" install --triplet=x64-linux \
  --x-feature=native-image --x-feature=native-audio --x-feature=native-video
export SKBUILD_BUILD_DIR="$PWD/build/python-release"
export SKBUILD_CMAKE_BUILD_TYPE=Release
uv pip install . --no-build-isolation \
  '-Ccmake.define.VANE_LOADABLE_EXTENSIONS=image;audio;video'
cmake --build "$SKBUILD_BUILD_DIR" --target vane_loadable_extensions
```

Select only the manifest feature and loadable target needed for a single
domain. Common FILE/AVIO implementation is linked internally; it is not a
fourth loadable extension. Optional artifacts stay outside the base wheel.
Use `scripts/build_extension_wheel.py` separately for each staged artifact.
The pinned vcpkg feature set disables FFmpeg default features and does not
select GPL, version3, or nonfree codecs. FFmpeg is [LGPL-2.1-or-later](https://ffmpeg.org/legal.html); zlib is
Zlib; DuckDB and extension sources are MIT. Package their copyright records,
Vane's LICENSE/NOTICE, and any transitive linked dependency notices explicitly.
The base license bundle must not be regenerated from an install tree that has
optional codecs merely because they are present there. For extension packages,
`scripts/sync_vcpkg_licenses.py --output <extension-notices.txt>` can generate
a separate complete installed-dependency notice bundle.

Static FFmpeg redistribution also requires corresponding source and a means
to relink the application with a modified FFmpeg, in addition to notices.
Extension release artifacts must include that source/build and relinking
material; this source PR does not publish binary wheels. Use the pinned vcpkg
baseline and recorded build configuration to reproduce codec inputs.

## Verify and measure

```bash
export VANE_TEST_NATIVE_IMAGE_EXTENSION="$SKBUILD_BUILD_DIR/vane_extensions/image.duckdb_extension"
export VANE_TEST_NATIVE_AUDIO_EXTENSION="$SKBUILD_BUILD_DIR/vane_extensions/audio.duckdb_extension"
export VANE_TEST_NATIVE_VIDEO_EXTENSION="$SKBUILD_BUILD_DIR/vane_extensions/video.duckdb_extension"
scripts/run_installed_pytest.sh tests/fast/test_native_media_extensions.py
```

The local artifact tests permit unsigned development artifacts on their own
connections. Distributed tests require signed installed providers and the
normal signature policy. Run `scripts/benchmark_native_media.py` from an
installed environment for repeatable Python/native timings; results identify
operations, inputs, rows, threads, repetitions, and the loaded artifact.
The harness records wall and process CPU times. Run `--backend python` and
`--backend native` in separate processes to compare peak RSS; the peak includes
imports, extension loading, and warmups and is not reset between repetitions.
The [measured workloads and reproduction guide](benchmarking/native_media/README.md)
include improvements and regressions; native execution is not uniformly faster.
