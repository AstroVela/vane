# Native image, audio, and video extensions

Vane provides three optional DuckDB C++ extensions: `image`, `audio`, and
`video`. Each is a separate `.duckdb_extension` artifact and can be installed,
loaded, and selected independently. The base runtime continues to provide
FILE, its media subtypes, IMAGE, Tensor, FILE field access/comparison, and
governed I/O. Loading an extension does not change those types or enable its
backend automatically.

`image_to_tensor` is a base C++ Image/Tensor storage conversion. It works with
either `image_backend` setting and requires no optional extension or Python
pixel helper. Its typed HWC result contract is documented in [IMAGE.md](IMAGE.md#image-to-tensor).

See [File Python values and media helpers](FILE_PYTHON_API.md) for immutable
value conversion, metadata results, and shared function/Expression options.

| Extension | Setting | Native operations |
| --- | --- | --- |
| `image` | `image_backend` | `image_file_metadata`, `decode_image_file`, `crop`, `resize`, `convert_image`, `encode_image`, `decode_image`, `image_hash` |
| `audio` | `audio_backend` | `audio_metadata`, `resample` |
| `video` | `video_backend` | `video_metadata`, `video_frames`, `video_keyframes`, `get_video_frame_by_idx`, `read_video_frames`, `build_video_index`, `video_index_info`, `video_scan_stats`, `VideoFrameSource` scanning |

Image cells materialize as UInt8, UInt16 or Float32 HWC NumPy arrays; both codec backends use the
same dynamic/fixed Image type and Arrow contract described in [IMAGE.md](IMAGE.md).

IMAGE pixel operators belong to the image extension's domain. Crop, resize,
color conversion and hashing accept all ten modes and operate directly on decoded pixels;
encoding supports PNG, JPEG, TIFF, GIF and BMP under the documented mode matrix;
their coordinates, result types, NULL rules and resource limits are documented
in [IMAGE.md](IMAGE.md). See
[VIDEO_FRAME_API.md](VIDEO_FRAME_API.md) for the Python/SQL streaming API.
The frame-list expressions, frame-index lookup, index construction and indexed
selection support both backends. Their explicit construction cost and complete
output contract are described in that guide.

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
`native_decode_image`, `native_image_hash`, `native_crop`, `native_resize`, `native_convert_image`, `native_encode_image`, `native_audio_metadata`,
`native_audio_resample`, or `native_video_metadata`.
Native video sources show `NATIVE_VIDEO_FRAMES`. Inspect the selected
setting with `current_setting('image_backend')`, and loaded artifacts with
`duckdb_extensions()`. Backend selection occurs when an expression is bound;
reusable prepared statements retain their bound implementation until rebound.
Lazy relations may be bound again when executed, and use the setting at that
binding. Set options before constructing and executing the query.

## Native contracts

The encoded-file operators call FFmpeg C libraries directly. Native crop uses
contiguous pixel copies; resize and color conversion use bounded C++ pixel
kernels; native PNG encoding uses zlib and TIFF uses libtiff. Other Image codecs
use FFmpeg directly. Native media execution does
not import Pillow, tifffile, imagecodecs, soundfile, soxr, or PyAV. Python result conversion and an
explicitly registered Python filesystem remain separate boundaries. Video follows
the shared selection, RGB, metadata and index contract in
[VIDEO_FRAME_API.md](VIDEO_FRAME_API.md); other media domains retain their own
numerical contracts. MIME validation
uses container families: MP4/MOV and Matroska/WebM respectively share a
native demuxer and accepted MIME family.
Absent content types, `application/octet-stream`, and `binary/octet-stream`
allow format detection. A matching domain wildcard (`image/*`, `audio/*`,
or `video/*`) also permits the detected format; a different domain is rejected.
Aliases for supported containers are normalized, including `image/x-png`,
`audio/mp3`, `audio/x-mp3`, `audio/aif`, `video/avi`, `video/mkv`, and
`video/x-m4v`. `application/ogg` accepts either an audio or video Ogg stream.

* Image supports PNG, JPEG, TIFF, GIF and BMP. Metadata reads headers without
  pixel decoding. Decode preserves 8/16-bit integer or Float32 RGB(A) depth when
  no output mode is requested; palette images expand to RGBA. Supported TIFF
  layouts, encoder modes, hash algorithms and byte limits are specified in
  [IMAGE.md](IMAGE.md). Unsupported content and MIME mismatches follow
  `on_error='raise'|'null'`; system and resource errors propagate.
* Audio supports WAV, AIFF, FLAC, MP3, AAC, Ogg, MP4, and WebM containers with
  decoders in the pinned FFmpeg build. For formats using libsndfile below,
  metadata matches Python SoundFile's format/subtype identifiers, sample rate,
  channels, and frame count. For example, 24-bit FLAC reports `FLAC`/`PCM_24`,
  while WAVEX and RF64 retain their distinct container identifiers. Known
  counts include zero for empty audio and exclude encoder delay/tail padding
  according to the same decoder used by `resample`. An unknown frame count
  remains NULL. Additional FFmpeg codecs retain their format/codec identifiers
  and only report frames where PCM duration establishes the count. In either
  case, duration is `frames / sample_rate` when frames is known, otherwise NULL;
  an estimated container duration is not exposed as the waveform duration.
  Metadata and resampling validate optional WAV `codec` tags and Ogg
  Vorbis/Opus `codecs` declarations against the detected codec. Ogg `codecs`
  are also checked for generic MIME declarations. Quoted values, escapes,
  comments, and RFC 2231 continuations are accepted. Encoded codec parameters use ASCII, UTF-8, or
  Latin-1; other charsets are rejected. Conflicting, malformed, or unsupported
  codec declarations raise a format error. RFC 2231 encoded parameter values
  cannot be quoted strings.
  `resample` returns `TENSOR(DOUBLE, [NULL, NULL])`
  with each row shaped `(frames, channels)`. Mono retains a channel dimension
  of one, empty audio has zero frames, and NULL input returns a NULL Tensor.
  Samples use frame-major order. The target sample rate remains the argument;
  retain it separately when it is needed alongside the waveform.
  Both backends resample with SoXR HQ using interleaved float64 input/output.
  Native uses libsndfile for PCM/float WAV and AIFF, 8/16/24-bit FLAC, MP3,
  and Ogg Vorbis/Opus, matching Python SoundFile's decoder, sample conversion,
  encoder-delay handling, and tail trimming. Additional codecs and containers,
  including Ogg FLAC and 32-bit FLAC, use FFmpeg decoding with the stream's
  packet time base; libswresample only converts their sample format/layout at
  the original rate before SoXR.
  No Python codec package or helper participates in native execution.
  Native rates are 1..384000 Hz, channel counts are 1..64, and rate changes
  share Python's maximum 64:1 ratio. Both explicit backends return
  the same logical Tensor type; see [VARIABLE_TENSOR.md](VARIABLE_TENSOR.md)
  for its Arrow, UDF, shape, dtype, and NULL contracts.
  Both normalize the complete output to
  `ceil(decoded_frames * target_sample_rate / source_sample_rate)` frames,
  trimming or zero-padding the tail once after decoder padding has been
  removed. The count uses integer arithmetic and actual decoded frames,
  including unknown-length streams. The source rate comes from libsndfile or
  the first FFmpeg decoded frame, since a container rate hint may differ from
  the decoder rate (for example, 8 kHz Opus input carried in WebM).
  Padding consumes the row and batch
  output budgets. Library versions and platform-specific codec arithmetic can
  still affect the last bits; sharing an algorithm is not a cross-build
  bit-for-bit guarantee for lossy audio. Metadata opens the shared decoder
  within the FILE view and read budget; it does not decode the full waveform
  to establish an unknown frame count.
* Video supports MP4/MOV, Matroska/WebM, AVI, MPEG-TS, MPEG, and Ogg containers with
  decoders in the pinned build. Metadata preserves unknown values as NULL.
  Connection-bound VideoFrameSource relations return RGB IMAGE values in their
  `frame` column with either backend. Pixel buffers grow with the actual emitted
  frames. Standalone Python VideoFrameSource tasks retain their RGB Tensor schema;
  `source.schema` describes those tasks. The bound relation's
  source association, zero-based frame index, PTS/DTS, duration, time base,
  and keyframe flag come from the selected stream and decoded frames.
  Times are relative to stream start when known, otherwise zero. Windows
  include both endpoints. Timestamp discontinuities reset sampling targets.
  Sampling uses exact rational presentation times and the shortest decimal
  representation of public DOUBLE options, without an epsilon.

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
of the stream, including for late time windows. Both backends' frame expressions accept
`index`, and public `read_video_frames` accepts a corresponding `indexes` list.
`build_video_index` records a complete sequential decode once; subsequent
indexed selections verify source blocks and seek to recorded keyframes.
`video_index_info` reports index construction work and `video_scan_stats`
measures a fresh selection. Python implements these algorithms independently
through PyAV and requires no loaded native video extension. Non-seekable inputs are not materialized to
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
maximum: 64 MiB) and a 30-second cooperative deadline. Audio shares this
deadline across FFmpeg container inspection and libsndfile opening; changing
parsers does not restart the timer. JPEG marker scanning
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

Connection-bound VideoFrameSource and public `read_video_frames` scans enforce
the same hard `max_partition_bytes` payload budget in both backends, including
each row's FILE/provenance fields. Binding rejects a single row that exceeds it.
Standalone Python Tensor tasks use a soft batch target.
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
select GPL, version3, or nonfree codecs. The audio feature additionally selects
libsndfile (including FLAC, Vorbis, Opus, and MPEG support) and libsoxr from the
same pinned baseline. FFmpeg, libsndfile, and libsoxr are LGPL-2.1-or-later;
see [FFmpeg licensing](https://ffmpeg.org/legal.html). The linked libFLAC,
libogg, libvorbis, and Opus libraries use
[BSD-3-Clause](https://spdx.org/licenses/BSD-3-Clause.html). zlib is Zlib;
DuckDB and extension sources are MIT. Audio extension wheels built with this
feature set use `Apache-2.0 AND MIT AND LGPL-2.1-or-later AND Zlib AND BSD-3-Clause`
as their [PEP 639](https://peps.python.org/pep-0639/) `License-Expression`.
The image feature adds libtiff and libjpeg-turbo. Its extension wheel expression
is `Apache-2.0 AND MIT AND LGPL-2.1-or-later AND Zlib AND libtiff AND BSD-3-Clause AND IJG`.
Package their copyright records,
Vane's LICENSE/NOTICE, and any transitive linked dependency notices explicitly.
The base license bundle must not be regenerated from an install tree that has
optional codecs merely because they are present there. For extension packages,
`scripts/sync_vcpkg_licenses.py --output <extension-notices.txt>` can generate
a separate complete installed-dependency notice bundle.

Static redistribution of these LGPL libraries also requires corresponding
source and a means to relink the application with modified libraries, in
addition to notices.
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

`scripts/benchmark_native_media.py` also accepts multiple input files,
`--sample-rate`, `--image-mode`, `--concurrency`, `--transport http`, and
`--runner ray --installed-provider`. Python-only runs need no extension.
HTTP byte/request counters come from a separate loopback server process.
Driver CPU/RSS exclude that server and Ray Workers; report those scopes when
interpreting the results. `--diagnostics` executes an additional local pass
after timings and RSS capture to inspect Python temporary-spool writes and
native audio phase costs. See the [validation guide](benchmarking/native_media/VALIDATION.md)
for the matrix and measurement boundaries.

The audio extension provides an explicit diagnostic function:

```sql
SELECT native_audio_resample_profile(audio_file('sample.wav'), 16000);
```

It accepts the same positional limits as `resample`, runs the same
native decoding and resampling implementation, and allocates the same bounded
waveform batch. It returns counters instead of the waveforms. This explicit
native function requires the loaded audio extension; regular resampling does
not enable diagnostic timers.

`setup_seconds` covers FILE opening, container inspection, and libsndfile
opening for supported audio; `decode_seconds`
covers decoder opening, packet reads, and decoded frames, including EOF;
`resample_seconds` covers resampler initialization and conversion, including
writing samples directly into the result buffer. `allocation_seconds` covers
reserving/growing that buffer. `file_read_seconds` measures successful
ResolvedFile read calls and overlaps setup/decode; do not add it again when
summing phase times. The phases exclude argument handling, bookkeeping,
diagnostic-result conversion, and destruction, so they do not sum to total
query latency.

`file_read_calls`, `file_bytes_read`, `decoded_frames`, `output_frames`, and
`output_bytes` describe each FILE execution. `buffer_growths` counts sample
buffer growth during that row; `buffer_capacity_bytes` is the retained
sample-vector capacity at the end of the row, including earlier rows in the
same engine batch. It is not process RSS or a per-row allocation.
`codec_version` is the linked FFmpeg libavcodec packed version;
`resampler_version` is the libsoxr packed version. `decoder_library` and
`decoder_version` identify the selected decoder library for that FILE;
`resampler_library='soxr_hq'` and `resampler_version_string` identify the
resampling configuration and linked runtime.
`source_sample_rate` is the decoded input rate actually used for resampling.
Each diagnostic batch starts with a fresh waveform workspace; normal execution
may reuse capacity across batches. Allocation counts therefore describe the
profiled invocation, not all uninstrumented allocator behavior.
NULLs, FILE windows, cancellation, and resource errors follow the same native
resampling contract. Profiling a large query can hit the same batch limit even
though its final diagnostic result is small.
