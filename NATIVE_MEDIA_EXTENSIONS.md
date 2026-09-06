# Native image, audio, and video extensions

Vane provides three optional DuckDB C++ extensions: `image`, `audio`, and
`video`. Each is a separate `.duckdb_extension` artifact and can be installed,
loaded, and selected independently. The base runtime continues to provide
FILE, its media subtypes, IMAGE, Tensor, FILE field access/comparison, and
governed I/O. Loading an extension does not change those types or enable its
backend automatically.

See [File Python values and media helpers](FILE_PYTHON_API.md) for immutable
value conversion, metadata results, and shared function/Expression options.

| Extension | Setting | Native operations |
| --- | --- | --- |
| `image` | `image_backend` | `image_file_metadata`, `decode_image_file` |
| `audio` | `audio_backend` | `audio_metadata`, `audio_resample` |
| `video` | `video_backend` | `video_metadata`, `video_frames`, `video_keyframes`, `get_video_frame_by_idx`, `read_video_frames`, `build_video_index`, `video_scan_stats`, `VideoFrameSource` scanning |

IMAGE pixel operators belong to the image extension's domain; this change
implements the encoded-file operations listed above. See
[VIDEO_FRAME_API.md](VIDEO_FRAME_API.md) for the Python/SQL streaming API.
The frame-list expressions and frame-index lookup support both backends. Native
indexed selection and its explicit construction cost are described in that guide.

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
Native video dispatch accepts the exact built-in VideoFrameSource. Selecting
native for a subclass raises an error before reading its files or executing
its custom tasks. Select Python explicitly when using a subclass's task/schema
contract.

The binder names native scalar functions explicitly in the plan. `EXPLAIN`
shows `native_image_file_metadata`, `native_decode_image_file`,
`native_audio_metadata`, `native_audio_resample`, or `native_video_metadata`.
Native video sources show `NATIVE_VIDEO_FRAMES`. Inspect the selected
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
Aliases for supported containers are normalized, including `image/x-png`,
`audio/mp3`, `audio/x-mp3`, `audio/aif`, `video/avi`, `video/mkv`, and
`video/x-m4v`. `application/ogg` accepts either an audio or video Ogg stream.

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
  Native VideoFrameSource returns RGB IMAGE values in its `frame` column.
  Pixel buffers grow with the actual emitted frames. Python VideoFrameSource
  tasks retain their RGB Tensor schema; `source.schema` describes those Python
  tasks, while the bound native relation exposes IMAGE. The native source's
  source association, zero-based frame index, PTS/DTS, duration, time base,
  and keyframe flag come from the selected stream and decoded frames.
  Times are relative to stream start when known, otherwise zero. Windows
  include both endpoints. Timestamp discontinuities reset sampling targets.
  Sampling tolerates a few DOUBLE rounding units at a threshold.

The video extension also registers bounded scalar frame-list, keyframe-list,
and exact-index functions. Public scalar calls normalize named/default SQL
arguments through macros, then bind to C++ scalar functions with an explicit
native or Python implementation. Scalar lists have per-row and per-chunk
payload limits; see [VIDEO_FRAME_API.md](VIDEO_FRAME_API.md#frame-expressions).

The video extension registers the `native_video_frames` table function used
by native VideoFrameSource, with IMAGE output in its `frame` column. The
video extension can produce IMAGE without loading the image extension.
These are extension execution entry points. Public `read_video_frames` uses
`native_read_video_frames` and returns both path and VIDEOFILE provenance with
fixed-shape IMAGE output in `data`. Its Python backend returns the same declared
types through a streaming DataSource.

Without a supplied index, exact global frame indices decode from the beginning
of the stream, including for late time windows. Native frame expressions accept
`index`, and public `read_video_frames` accepts a corresponding `indexes` list.
`build_video_index` records a complete sequential decode once; subsequent
indexed selections verify source blocks and seek to recorded keyframes.
`video_index_info` reports index construction work and `video_scan_stats`
measures a fresh native selection. Non-seekable inputs are not materialized to
temporary files. Unsupported random access propagates through the FILE reader.

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
maximum: 64 MiB) and a 30-second cooperative deadline. JPEG marker scanning
shares 64 KiB read buffers within the FILE view and charges all fetched bytes
to the budget; PNG metadata retains exact small header reads.
Decoding checks input-view size, cumulative reads, dimensions,
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
regardless of `read_task_count`. Native IMAGE output avoids fixed ARRAY pixel
reservations for unused vector rows, including empty scans. The payload budget
does not bound total process RSS. Codec contexts, reference frames,
conversion buffers, and downstream query state also consume memory.

For the older VideoFrameSource, native `max_partition_bytes` is a hard payload limit, including each row's
FILE/provenance fields. Binding rejects a single row that exceeds it. This
intentionally differs from Python VideoFrameSource's soft batch target. The
public `read_video_frames` API enforces a hard payload budget in both backends.
Audio/video metadata `max_bytes` controls
both the callback read budget and FFmpeg's format/stream probe size, up to
64 MiB. Decode operations retain their separate 8 MiB probing limit.

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
