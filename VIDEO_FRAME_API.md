# Streaming video frames

## Inspecting media streams

`vane.probe_media(file, connection=con)` is an eager Python value API for bounded
audio/video container inspection. It accepts `File`, `AudioFile` and `VideoFile`,
including logical byte ranges, and requires the `video` extra. It can also run
inside a public Vane UDF. It does not load a model or select an ingest policy.

```python
probe = vane.probe_media(vane.File("clip.mkv"), connection=con)
for stream in probe.streams:
    print(stream.index, stream.kind, stream.codec, stream.language, stream.default)
```

The immutable `MediaProbe` reports format names, optional container duration and
ordered `MediaStream` records. Stream times are in seconds on the container
timeline, with the original rational time base retained. Video dimensions, audio
sample rate/channels and default/forced/attached-picture dispositions are
reported when applicable. Unknown values remain `None`. Declared language does
not establish spoken language; an audio stream does not establish speech and a
subtitle stream does not establish decoder support. Burned-in captions require
separate visual processing.

Detection inspects bytes rather than trusting the filename or MIME label.
`max_bytes` defaults to 8 MiB (maximum 64 MiB); `buffer_size` defaults to 64 KiB.
The probe reuses video metadata's range/read limits, bounded parser options and
nested-I/O rejection. Errors use `VideoFileFormatError`/`VideoFileLimitError`;
original I/O errors propagate. Exhausted probes never return a partial inventory.

## Reading frames

`vane.read_video_frames` and SQL `read_video_frames` emit one row per selected
frame. The `data` column contains decoded `IMAGE('RGB', H, W)` values. Python
fetches materialize them as UInt8 HWC NumPy arrays (`vane.Image` is a typing
alias). Arrow uses the `vane.image` extension over fixed-size UInt8 lists.
The native_media extension includes both image and video processing.

Install `vane-ai[video]` for the Python backend and its decoder and memory
admission dependencies. The native backend uses the optional native_media extension.

```python
import vane

con = vane.connect()
frames = vane.read_video_frames(
    "clip.mp4", 224, 224,
    start_time=1, end_time=5, sample_interval_seconds=0.5,
    connection=con,
)
frames.select("path", "frame_index", "frame_time", "data").show()
```

```sql
SELECT path, file, frame_index, frame_time, data
FROM read_video_frames(
    'clip.mp4', 224, 224,
    start_time => 1, end_time => 5, sample_interval_seconds => 0.5
);
```

The first three arguments are the source, output height, and output width.
The fourth optional positional argument is `is_key_frame`, which can also be
named. The remaining options are named arguments in both APIs. Python also
accepts `connection` to select the query session.

## Input and backend

Inputs are exact path strings, FILE/VIDEOFILE values, or lists of these values.
Python additionally accepts string-valued `os.PathLike` objects. FILE views
retain all five fields, including `position`, `size`, and `checksum`. URL
strings become whole-object VIDEOFILE references without inspecting storage.
Directory and glob discovery is separate from this function. A top-level
NULL/None or empty list returns an empty table with the declared schema. NULL
list elements and other media subtypes are rejected.

The connection's `video_backend` setting defaults to `python`. The C++ SQL
binder selects a Python DataSource whose Worker tasks use PyAV. To select the
native FFmpeg C++ scan, install and load the matching native_media extension, then set
the backend before constructing and executing the query:

```python
vane.load_installed_extension("native_media", connection=con)
con.execute("SET video_backend = 'native'")
frames = vane.read_video_frames("clip.mp4", 224, 224, connection=con)
```

Native execution calls the native_media extension directly. An unavailable native
extension fails during binding; no automatic fallback is performed. Binding
validates arguments and builds tasks without opening the videos. `EXPLAIN`
shows `DATASOURCE_SCAN` for Python and `NATIVE_READ_VIDEO_FRAMES` for native.
Prepared statements retain their bound implementation; lazy relations may
bind again when executed.

Both paths read the FILE's logical byte window through the executing query's
connection or Worker context. Credentials are resolved there and are never
copied into the source values or serialized tasks. Distributed native execution
requires the same trusted extension provider on every Worker, as described in
[DISTRIBUTED_EXTENSIONS.md](DISTRIBUTED_EXTENSIONS.md).

## Output

| Column | Type | Meaning |
| --- | --- | --- |
| `path` | VARCHAR | Source URL |
| `file` | VIDEOFILE | Original logical FILE view |
| `frame_index` | BIGINT | Zero-based presentation-order decoded frame index |
| `frame_time` | DOUBLE | Seconds relative to stream start (zero origin if the start is unknown) |
| `frame_time_base_numerator` | BIGINT | Numerator of the frame timestamp unit |
| `frame_time_base_denominator` | BIGINT | Denominator of the frame timestamp unit |
| `frame_pts` | BIGINT | Presentation timestamp in time-base units |
| `frame_dts` | BIGINT | Decode timestamp, when exposed by the decoder |
| `frame_duration` | BIGINT | Duration in time-base units, when known |
| `is_key_frame` | BOOLEAN | Decoder keyframe flag |
| `data` | IMAGE('RGB', H, W) | Packed RGB uint8 pixels resized to the requested dimensions |

Unknown temporal metadata remains NULL. Time windows include both endpoints;
sampling selects frames when their timestamps reach the next interval target.
Both backends capture the stream origin when opening the input, using zero
when it is unknown. Estimates discovered while reading packets do not shift
that origin. Index construction, scalar/list expressions and streaming sources
use this same rule.
Without an explicit index, exact frame indices use sequential decoding from
stream start, including for late time windows. Both backends also accept
reusable indexes for keyframe seeking, as described below. There is no
temporary-file materialization fallback. With the same FFmpeg component versions
and decoding configuration, both backends follow the same complete frame contract,
including DTS and zero-valued frame duration. Sampling compares exact rational
presentation times against the shortest decimal representation of public DOUBLE
options, with no epsilon. Frame indices remain presentation-order ordinals after
keyframe filtering; `is_key_frame=False` excludes keyframes.

Both backends resize and convert decoded video frames directly to RGB with
FFmpeg's frame-aware scaler, bilinear interpolation, and one scaling thread.
The conversion retains the source YUV matrix, range, and chroma location;
transfer-function and color-primary conversion is not requested. Vane requests
resizing and RGB conversion together in one scaler call. Scalar expressions,
streaming reads, and indexed reads share this pixel conversion policy. Regression tests
require byte-identical RGB output between backends for the supported test
fixtures, including odd dimensions, full/limited-range color, 10-bit video,
and interlaced frames. Interlaced field layout is retained during scaling.
Both backends enable FFmpeg's `GENPTS` demuxer flag, which can recover missing
presentation timestamps through packet lookahead, including for AVI. They probe
before decoding, with the same probe limits, `skip_frame=all` probe policy and
decoder packet time base. They decode normally
and then apply keyframe filters. Indexed reads validate the decoded frame and
return the original sequential decode's recorded DTS and duration. Vane does not
copy PTS into DTS or discard timestamps to force parity. Results from different FFmpeg releases require
separate validation.

Rows from independent file tasks have no global order. Use `ORDER BY` when
consuming ordered results. An explicit `frame_limit` creates one ordered task
that visits input files in list order, enforcing a single global output limit.
Duplicate inputs remain duplicate work.

## Options and resource limits

| Option | Default | Contract |
| --- | --- | --- |
| `is_key_frame` | NULL | TRUE selects keyframes; FALSE selects other frames |
| `start_time` | 0 | Finite nonnegative seconds |
| `end_time` | NULL | Inclusive upper bound, at least start_time |
| `sample_interval_seconds` | NULL | Finite positive sampling interval |
| `frame_limit` | NULL | Nonnegative global output limit; zero performs no I/O |
| `read_task_count` | NULL | Positive number of balanced file groups, capped by input count; defaults to one per file |
| `on_error` | 'raise' | 'raise' or 'skip'; skip applies only to encoded format errors |
| `max_input_bytes` | 8 GiB | Encoded logical view limit; maximum 16 GiB |
| `max_decoded_frames` | 1,000,000 | Per-file decoding limit, including filtered frames; maximum 100,000,000 |
| `max_pixels` | 32 Mi pixels | Input and output frame limit; maximum 32 Mi pixels |
| `max_partition_bytes` | 10 MiB | Hard emitted batch payload budget; maximum 256 MiB |

Height and width must be positive integers no greater than 100,000; their
product must fit `max_pixels`. Input metadata is limited to 100,000 FILE views
and 64 MiB. The batch budget reserves RGB bytes plus FILE, duplicated path,
and temporal fields. A row that cannot fit is rejected during binding. Both
scan backends allocate decoded pixel payloads for actual rows, including short
final batches. After scanning, fixed Image columns use dense engine ARRAY
vectors; their batch capacity and downstream operators also contribute to query
memory, beyond the decoder partition budget.

The payload budget is not a total-process RSS limit. Codec reference frames,
conversion buffers, temporary Arrow copies, and downstream query operators
consume additional memory. The scan streams bounded batches, while collecting
all rows into a Python list or sorting the complete result adds its own memory
cost. Cancellation is cooperative around I/O and decoding. I/O, permission,
resource, allocation, and cancellation failures propagate under both error
policies.

## Fixed-shape IMAGE types

Combining different IMAGE shapes with the same known mode in `CASE`, `VALUES`,
`UNION`, `COALESCE`, or list construction yields `IMAGE(mode)`. Different or
unknown modes yield generic `IMAGE`. This also applies within nested types.
Equal fixed layouts retain their constraint. Narrowing a mode or shape
requires an explicit validated cast.

`vane.image_type('RGB', H, W)` and `IMAGE('RGB', H, W)` use a fixed-size UInt8
pixel array. `IMAGE('RGB')` constrains only the mode and uses dynamic STRUCT
storage with a pixel list. Fetched Image cells are HWC UInt8 NumPy arrays;
Arrow carries their mode and dimensions with the `vane.image` extension type.
See [Decoded Image values](IMAGE.md) for the complete storage and value contract.

An explicit IMAGE-to-IMAGE cast checks shape; `TRY_CAST` returns NULL for a
layout mismatch. Casts do not decode, resize, or change pixels. Python typed
values, row and batch UDF outputs, and distributed exchange validate the shape
as well as IMAGE's existing field contract. Nested declared types retain the
constraint. Ordinary raw STRUCT values cannot acquire IMAGE semantics by cast.

Fixed IMAGE layout narrowing requires an explicit `CAST`/`TRY_CAST` or
`Expression.cast`, including IMAGE leaves in LIST, ARRAY, MAP, and STRUCT.
`INSERT` and `UPDATE` do not implicitly constrain generic or differently sized
IMAGE values. Each explicit cast validates the actual mode, height, and width;
`TRY_CAST` replaces a mismatched IMAGE leaf with NULL. If converted MAP keys
become NULL or duplicate, it nulls that MAP. Python declared values
and UDF output types validate their declared layout at the conversion boundary.
The explicit validation mode is retained when a plan is sent to a worker.
Nested casts ignore child storage beneath NULL containers and inactive UNION
members. IMAGE-bearing casts retain their validation errors during optimizer
filter rewrites, including converted MAP keys whose IMAGE layout widens.
Column pruning can push down field extraction while keeping these validated
casts in the expression. In particular, casting fields of unnested frames
retains IMAGE layout checks and MAP key validation before values reach storage.

## Frame expressions

`video_frames`, `video_keyframes`, and `get_video_frame_by_idx` consume VIDEOFILE
expressions in Python and SQL. `expr.video_frames(...)` and
`expr.video_keyframes(...)` accept the same keyword options as the corresponding
Python functions. These functions preserve one output row per input video;
`read_video_frames` emits separate rows and bounds each emitted batch.

```python
videos = con.sql("SELECT video_file('clip.mp4') AS file")
videos.select(
    "file",
    vane.video_frames(vane.col("file"), start_time=1, end_time=2,
                     width=224, height=224, sample_interval_seconds=0.5).alias("frames"),
    vane.get_video_frame_by_idx(vane.col("file"), 10).alias("frame_10"),
).show()
```

```sql
SELECT file,
       video_frames(file, start_time => 1, end_time => 2,
                    sample_interval_seconds => 0.5) AS frames,
       video_keyframes(file, end_time => 2) AS keyframes,
       get_video_frame_by_idx(file, 10, on_error => 'null') AS frame
FROM videos;
```

| Function | Result |
| --- | --- |
| `video_frames(file, ...)` | LIST of frame records: `file: VIDEOFILE`, the eight temporal fields in the streaming schema, and `data: IMAGE('RGB')` |
| `video_keyframes(file, ...)` | LIST of `IMAGE('RGB')` values selected with the decoder's keyframe flag |
| `get_video_frame_by_idx(file, idx, ...)` | One `IMAGE('RGB')` at the zero-based presentation-order decoded index |

Frame records retain the complete source FILE view. For image-only keyframe
and index results, retain the input FILE column alongside the result when
source association is needed. The scalar images use dynamically shaped,
mode-constrained `IMAGE('RGB')` values with UInt8 pixels. Their values carry
the actual dimensions, including after an explicit resize.

Both list functions accept `start_time=0`, `end_time=None`, `width=None`,
`height=None`, and `sample_interval_seconds=None`. `video_frames` additionally
accepts `is_key_frame=None`. Width and height must be supplied together; when
omitted, each selected frame keeps its decoded dimensions. Time selection,
sampling, and exact indices follow the streaming contracts above. Frame-index
lookup closes the decoder after returning the requested frame. All three
functions accept `index=None`; a supplied BLOB selects indexed access.

All three functions accept `on_error='raise'|'null'`, `max_input_bytes=8 GiB`,
`max_decoded_frames=1,000,000`, `max_pixels=32 Mi pixels`, and
`max_output_bytes=64 MiB`. The two list functions also accept
`max_output_frames=10,000`. SQL supports positional and named arguments;
Python selection and resource options are keyword-only and can be Expressions.

The output limit counts RGB pixels and frame metadata for one input row. It
may be raised to at most 256 MiB and 100,000 selected frames. Each scalar
execution chunk also has a 256 MiB output payload ceiling. Allocations made
before a format failure still count toward that ceiling. Exceeding a limit
raises a resource error instead of returning a truncated or NULL list; use
`read_video_frames` for results that need to stream past these limits. Input,
decoded-frame, and pixel ceilings match the streaming API.

NULL VIDEOFILE inputs return NULL. A successful selection with no matches
returns an empty list. A NULL frame-number argument (`idx`) or required scalar
option returns NULL. A NULL seek index (`index`) selects sequential execution.
NULL end time, dimensions, keyframe filter, or interval mean no restriction.
Missing frame indices raise an index-range error, or return NULL under the
explicit null policy. Negative indices and invalid options always raise.
Only encoded-format errors may become NULL under `on_error='null'`; I/O,
dependencies, resource limits, cancellation, and unexpected system failures
remain errors. A format error after partial decoding nulls the complete row.

The explicit `video_backend` option applies to all three functions. Python
execution enters PyAV through a C++ scalar bridge using the executing query's
FILE context. Native execution calls the loaded native_media extension's FFmpeg C++
operators. Binding and expression construction do not open files. Worker
credentials resolve against each original FILE URL and logical byte window;
the Python execution token expires when the scalar row finishes and is never
serialized. No automatic backend or materialization fallback is provided.

## Reusable video indexes

`build_video_index(file)` uses the selected backend and returns an opaque BLOB recording the actual
presentation-order frames and their keyframe anchors. Materialize this value
once and store it alongside its VIDEOFILE before issuing repeated selections:

```sql
CREATE TABLE indexed_videos AS
SELECT file, build_video_index(file) AS seek_index FROM videos;

SELECT get_video_frame_by_idx(file, 1200, index => seek_index),
       video_frames(file, start_time => 50, end_time => 51,
                    sample_interval_seconds => 0.5, index => seek_index)
FROM indexed_videos;
```

The Python functions accept the same index through `index=bytes_or_expression`;
`expr.video_frames(...)` and `expr.video_keyframes(...)` share these options.
For streaming output, `read_video_frames(..., indexes=[index, ...])` takes one
non-NULL BLOB for each input FILE, in the same order. SQL uses the named
`indexes => LIST<BLOB>` argument. Ray splits transport each FILE with its index
and open the source on the executing Worker using its existing FILE context.

Building an index performs a complete content-hashing pass and a complete
sequential decode. It does not store decoded pixels or encoded video content.
Index construction is an explicit cost; putting the builder in an unmaterialized
expression can repeat that work. Indexed queries select frames using recorded
timestamps and exact frame numbers, seek backwards to a recorded keyframe, and
decode through the selected frame. Sparse selections can skip intervening GOPs.
No frame number is inferred from average FPS. Returned metadata retains the
original decode's frame index, PTS/DTS, duration and source association.

Indexes require a known time base, unique strictly increasing presentation
timestamps, and an initial keyframe. Variable frame rates, nonzero timestamp
origins and B frames are supported within that contract. Duplicate/missing
timestamps and discontinuities are explicitly unsupported for indexing. Every
frame returned after seeking is checked against its indexed decoded content;
a seek that cannot reproduce the indexed frames raises an unsupported-access
error, including unsafe keyframe/open-GOP behavior. There is no sequential
retry after an indexed access fails. Omitting the index explicitly selects the
sequential path, which remains available with either backend. SQL expressions and
`read_video_frames` accept indexes built by either backend. The Python backend
implements construction, index parsing, content verification, seeking and frame
selection in Python through PyAV. It requires no loaded native_media extension. Native
execution implements the same algorithm in C++ through FFmpeg. Both use governed
FILE I/O and the base engine's canonical serialization of the open FILE view and
its resolved metadata. Python value iterators retain their sequential API.

An index belongs to one immutable FILE view. It binds all five FILE fields,
resolved source metadata, the engine SourceID, and FFmpeg component versions.
Source metadata or runtime identity changes require rebuilding it. Each logical
64 KiB block actually read is verified against the index before its bytes reach
the codec; the final block stops at the FILE window boundary. This verifies
accessed content, rather than re-reading the entire source on every selection.
The index's SHA-256 detects accidental index corruption; it is not a signature
or authorization token. Preserve indexes as application data from their trusted
builder. FILE access always requires the current executing connection's
permissions. No credentials, filesystem handles or Python objects are encoded.

`build_video_index` accepts the existing `max_input_bytes=8 GiB`,
`max_decoded_frames=1,000,000`, `max_pixels=32 Mi pixels`, and
`max_index_bytes=64 MiB` limits. The index limit may only be lowered. Index BLOB
output is limited to 256 MiB per scalar chunk; construction also retains bounded
frame metadata and content digests while encoding that row. Individual input
indexes are capped at 64 MiB. Reading retains a bounded BLOB copy and decoded
index records in addition to the input vector, one 64 KiB verification buffer,
and the existing codec/output buffers. A streaming source's
64 MiB metadata ceiling includes all supplied indexes and FILE descriptions.
Index parsing, hashing and selection check cancellation; codec calls retain
their existing cooperative cancellation boundary.

With an index, `max_decoded_frames` limits frames returned by the decoder during
that selection, including discarded seek preroll. It does not limit the global
requested frame number. The index builder has its own full-decode frame budget.
Streaming `on_error='skip'` retains the valid decoded prefix before a content
error, including a final partial batch, then continues with the next FILE. SQL
streaming preserves the public error category across the Python Arrow boundary.
Index format, identity, integrity, seek-reproduction and resource failures are
not suppressed by `on_error='null'` or `'skip'`. Ordinary encoded-format failures
retain the existing error policy. There is no automatic materialization.

`video_index_info(index)` inspects a BLOB without opening a FILE and returns
`frame_count`, `keyframe_count`, `source_bytes`, `index_bytes`,
`build_bytes_read`, and `codec_version`. `vane.video_index_info` constructs its
expression. Its implementation follows `video_backend`, including on connections
that have not loaded the native_media extension.
Both parsers require `source_bytes <= build_bytes_read <= source_bytes + 64 GiB`:
construction hashes the FILE once, then allows decoder reads up to four times the
public 16 GiB input limit. This is a global consistency bound; the original
construction input limit is not serialized, and the checksum is not authentication.

`video_scan_stats(file, ..., index=None, idx=None)` runs selection and
returns `bytes_read`, `decoded_frames`, `seeks`, and `selected_frames`. It accepts
the same time/keyframe/sampling options; a non-NULL `idx` selects a single exact
frame instead. It is a separate diagnostic execution with no output images.
Indexed reads still convert unscaled RGB pixels to verify decoded content.
Bytes include verification
block reads; frames count decoder outputs including discarded preroll, excluding
codec-internal lookahead. Both sequential and indexed measurements use the same
cursor as their backend's expressions and streaming source. The
[benchmark guide](benchmarking/video_seek/README.md) separates index construction
from repeated query latency and includes loopback HTTP response-byte accounting.

## Metadata and index representation

`video_metadata` returns `width`, `height`, `fps`, `duration`,
`container_duration`, `frame_count` and `time_base`. `duration` describes only the
selected video stream; `container_duration` describes the entire container,
including longer audio streams. Unknown/zero duration sentinels are NULL.
Unknown frame counts remain NULL and are never estimated from duration and FPS.
`video_index_info(build_video_index(file)).frame_count` supplies an exact decoded
count when an application needs one. A missing FILE returns NULL in scalar
expressions; a valid frame selection with no matching frames returns `[]`.

Index format version 2 begins with `VVIDX002`. All integers are little-endian
64-bit fields. Its 160-byte header contains the libavcodec/libavformat/libavutil/libswscale
version integers, logical source size, SHA-256 of the engine SourceID, SHA-256
of the canonical FILE/source-metadata serialization, time-base numerator and
denominator, signed timestamp origin, build bytes read, block count and frame
count. Next come the SHA-256 digests of consecutive 64 KiB source blocks. Each
88-byte frame record contains signed PTS, DTS and duration, width, height, pixel
format, keyframe flag and a 32-byte frame digest. Unknown DTS uses INT64_MIN.
The nearest preceding keyframe ordinal is derived while parsing. A trailing
SHA-256 covers every preceding index byte.

A frame digest covers width, height and source pixel format as 64-bit integers,
followed by tightly packed, unscaled RGB bytes from the public video converter.
It excludes allocator padding. Original encoded content is independently checked
by source block digests. Builders and readers use this same procedure in both
backends. Version 1 indexes must be rebuilt; there is no alternate parsing path.

## Python video embedding orchestration

`vane.ai.VideoEmbeddingPipeline` accepts a video request dictionary and processes
fixed clips. Channel selection and vector output are independent:

| Field | Values and meaning |
| --- | --- |
| `input` | `type="video"`, `uri`; optional `asset_id` and immutable `source_version` |
| `modalities` | Nonempty, unique list of `visual`, `audio`, `subtitles`, `transcription`, `ocr` |
| `selection.policy` | `all` (default), or `first_available` within each clip |
| `selection.priority` | Only with `first_available`; an exact permutation of the modalities. Defaults to their supplied order. |
| `selection.unit` | `clip` only |
| `segmentation.clip_seconds` | Finite positive seconds; default 10 |
| `output` | `separate` (default), `fused`, or `both` |
| `return_timeline` | Include selected content intervals and text; default `true` |
| `async` | Return a task handle immediately; durable when configured with `VideoJobStore`; default `false` |
| `sync_budget` | `timeout_seconds` and `max_video_seconds`, each defaulting to 60 |

These controls do not implement search AND/OR. Independent vectors can be used
for either recall unions or intersections; a fused vector does not guarantee
that every modality matches. Configure retrieval separately. `auto` is rejected
until a versioned automatic policy is defined.

For example, prepare an explicitly versioned visual profile and submit a task:

```python
from vane.ai import VideoEmbeddingPipeline, VideoEmbeddingProfile

visual = VideoEmbeddingProfile.from_provider(
    "cosmos-reviewed-revision-v1",
    modality="visual",
    provider="transformers",
    model="nvidia/Cosmos-Embed1-224p",
    revision=reviewed_model_commit,
    device="cuda",
    dtype="float16",
    trust_remote_code=True,
)
with VideoEmbeddingPipeline(profiles={"visual": visual}) as pipeline:
    task = pipeline.create({
        "input": {
            "type": "video", "uri": "s3://bucket/video.mp4",
            "asset_id": "video-123", "source_version": "object-version-1",
        },
        "modalities": ["visual"],
        "selection": {"policy": "all", "unit": "clip"},
        "segmentation": {"clip_seconds": 10},
        "output": "separate",
        "return_timeline": True,
        "async": True,
    }, resources={"embed": {"gpus": 1}})
    print(task.id)  # Available before container probing or model loading finishes.
    result = pipeline.get_task(task.id).result()
```

The complete runnable visual example is
[`examples/video_embedding_pipeline.py`](examples/video_embedding_pipeline.py).
Install `vane-ai[video]` plus the chosen model's extras, such as `cosmos` or
`clap`. Profiles reuse existing Vane descriptors and typed embedding validation.
`create()`, `ingest()` and `batch_ingest()` compile the same Ray execution plan.
The first returns a vector response (or an asynchronous task); the ingestion
methods return a lazy relation for one final sink. There is one implementation
of selection, preprocessing, encoding, assembly and fusion. `create()` accepts
the same `resources` and `max_batch_bytes` keyword options as ingestion. Models
load lazily in UDF workers and actors retain them across batches. Model revisions,
preprocessing options and normalization belong in a versioned profile name.

`VaneVideoSource` (from `vane.ai.video`) uses Vane FILE access for local and remote
URIs and PyAV for extraction. It probes the container once for both selection
policies, then uses the first non-attached video stream, first audio stream and
first subtitle stream. Video sampling defaults to eight distinct frames per
clip; audio is windowed to 48 kHz mono PCM. Cosmos requires a CUDA GPU on its Ray worker.
Configure `frames_per_clip` and
`sample_rate` for the selected profiles. Decoder and model limits remain errors;
a short clip that cannot satisfy a model's frame count is not padded implicitly.
The container start timestamp defines a common origin for all tracks. Audio gaps
retain their position within a clip. Text subtitles crossing a boundary contribute
to both clips; point events belong only to the half-open interval containing them.
Inputs must report a finite positive duration; live or unknown-length sources are
rejected before segmentation.

Speech recognition and OCR support built-in adapters or application extractors. Pass
`VaneVideoSource(extractors={"transcription": asr, "ocr": ocr})` as `source=`.
Each callable takes `(uri, ClipWindow, VideoMediaInfo)` and returns a sequence of
`TimedContent(start, end, value)` objects. Timestamps are absolute asset seconds,
not clip-relative seconds; text channels carry strings. ASR can obtain bounded
PCM through `source.extract(uri, "audio", clip, media)` and OCR can obtain frames
through the `visual` channel. An override for `subtitles` can supply external
captions or a language-selected track. Bitmap subtitles require such an explicit
OCR override. A custom `VideoSource` can replace probing and extraction entirely.

Return `[]` for normal absence. Whitespace-only text and content outside the
current clip are empty for selection. Missing tracks and empty ASR may advance
`first_available`; invalid timestamps, decoding errors, timeouts, invalid vectors
and model failures propagate. There is no error-to-absence conversion or automatic
retry in this orchestration layer. The no-retry contract also fences replay of
outer native execution fragments, so a failed UDF cannot be rerun indirectly.

For clip search, keep visual independent and add a dialogue priority group:

```python
request = {
    "input": {
        "type": "video", "uri": "s3://bucket/video.mp4",
        "asset_id": "video-123", "source_version": "object-version-1",
    },
    "modalities": ["visual", "subtitles", "transcription", "ocr", "audio"],
    "selection": {
        "policy": "all",
        "unit": "clip",
        "groups": [{
            "modalities": ["subtitles", "transcription"],
            "policy": "first_available",
            "priority": ["subtitles", "transcription"],
        }],
    },
    "segmentation": {"clip_seconds": 10},
    "output": "separate",
    "async": True,
}
```

Configure a profile for every requested independent channel, including both text
priority members. Groups must be disjoint subsets of the requested modalities
and require outer `all`; ungrouped channels execute independently. With a global
`first_available` priority of transcription, visual, audio, nonempty ASR really
does skip visual for that clip.

`fused` and `both` require a `VideoFusionProfile` with an explicit provider, model,
profile name, dimensions and supported modality combinations. Its `embed` callback
receives `(clip, selected_timed_content, fusion_options)` and may return a vector
or an awaitable vector. It must use a model capable of joint inputs and interpret
the supplied common timeline. The built-in Qwen3-VL and Cohere adapters below
encode joint inputs. Vane never adds vectors merely because their dimensions
agree. Unsupported combinations after selection fail, including a missing input
or a single selected modality unless that combination is explicitly supported.
An entirely empty clip produces no vector and retains its audit record.
`fusion.method` must match the configured model; `fusion.weights` requires explicit
model support, must cover requested channels, and passes only selected weights
to the callback. A future shared-space fusion adapter must separately establish
compatibility of the exact encoding profiles.

Responses contain `data` vectors with `clip_id`, `asset_id`, `source_version`,
`uri`, `start_sec`, `end_sec`, actual `modality`,
contributing `modalities`, `provider`, `model` and `profile`. The `clips` array
retains every clip, even when empty, with a decision for every requested channel:
`all`/`first_available` for selection, or `no_stream`, `no_content`,
`lower_priority` for skipping. Lower-priority channels are not extracted, so this
reason does not assert whether they would have contained content.
`return_timeline=false` omits detailed content intervals/text while retaining
vector time ranges and all selection decisions.

Task states are `pending`, `processing`, `ready`, `failed`, `cancelling` and
`cancelled`. `task.result(timeout=...)` raises the execution error or a caller wait
timeout; that wait timeout alone does not cancel the task. `task.cancel()` requests
cooperative cancellation. A synchronous request exceeding its wall-clock budget
raises `VideoEmbeddingTimeout` with `task_id` and interrupts the owned Vane query.
The runner cancels outstanding work and tears down its resources; cancellation
may race with work already admitted. Source I/O also has its own timeout.
`close()` and context exit wait for submitted jobs and query cleanup. The query
connection belongs to that task, so cancellation does not interrupt another
caller's connection. The video-duration budget is checked in the common probe
stage before any extraction. The wall-clock budget includes queueing, plan
construction and Ray startup.

Without a job store, task IDs are process-local. The
pipeline submits one `create()` job at a time, with stages running in parallel
within the query, retains at most `max_tasks=128` tasks, and bounds
segmentation with `max_clips=10000`. Use `forget_task(id)` after consuming a
completed task to release its result; running tasks cannot be forgotten.

### Built-in local and cloud video models

Model adapters load only inside executing workers. `VaneVideoSource(asr=..., ocr=...)`
decodes bounded audio/frames and passes them to the selected adapters. No application
callback is required. Every entry point uses the same adapters. For example:

```python
from vane.ai.video import VaneVideoSource
from vane.ai.video_models import (
    FasterWhisperASR, RapidOCRExtractor, Qwen3VLFusion,
    DeepgramASR, GoogleVisionOCR, CohereFusion,
)

local_source = VaneVideoSource(
    asr=FasterWhisperASR("/models/whisper", profile="whisper-checkpoint-v1", language="zh"),
    ocr=RapidOCRExtractor(
        "/models/ocr/det.onnx", "/models/ocr/rec.onnx", "/models/ocr/keys.txt",
        profile="ocr-checkpoint-v1",
    ),
)
local_fusion = Qwen3VLFusion("/models/qwen", frame_size=224).profile("qwen-joint-224-v1")

cloud_source = VaneVideoSource(
    asr=DeepgramASR(profile="deepgram-v1", model="nova-3", language="en"),
    ocr=GoogleVisionOCR(profile="vision-ocr-v1"),
)
cloud_fusion = CohereFusion(dimensions=1024).profile("cohere-joint-v1")

pipeline = VideoEmbeddingPipeline(
    profiles=embedding_profiles, source=cloud_source, fusion=cloud_fusion,
)
```

Install `vane-ai[video,video_asr,video_ocr]` for local ASR/OCR and `qwen3_vl` for
local fusion. Checkpoints, ONNX files and the OCR character dictionary must already
exist on every worker; adapters do not download models. RapidOCR uses CPU ONNX
without an angle classifier. Whisper explicitly selects CPU/int8 or CUDA/precision
and uses one decoding temperature. Configure stage resources to match the models;
actors retain loaded models between batches and tasks release them per invocation.

Qwen fusion reuses the existing BF16 CUDA checkpoint contract, encodes the frames
and timestamped text in one forward pass, and resizes RGB frames to the declared
square `frame_size` with bicubic interpolation. It preserves frame times/counts;
the existing even-frame and token/byte limits still apply. Include this preprocessing
in the profile version. Cohere sends one Embed v4 input containing PNG frames,
their absolute times, and timed text, with truncation disabled. Both adapters support
visual and the three text channels, including their nonempty subsets. Neither accepts
raw audio or fusion weights; unsupported selected combinations fail without dropping inputs.
Use independent audio embeddings for audio-inclusive indexing. These capabilities
do not establish retrieval quality on a particular corpus.

Cloud adapters read `DEEPGRAM_API_KEY`, `GOOGLE_VISION_API_KEY` and `COHERE_API_KEY`
from the worker environment (variable names are configurable). They send only the
selected decoded clips/frames to their respective providers. Credentials are not
captured in descriptors or job records. Requests have finite timeouts and bounded
bodies, disable redirects and retries, and propagate errors. Empty successful ASR/OCR
results remain normal absence. ASR offsets are rebased to absolute asset seconds;
OCR observations use their sampled frame times. Channel metadata includes the
extraction provider/model/profile in addition to the embedding identity.

Protocol references: [faster-whisper](https://github.com/SYSTRAN/faster-whisper),
[RapidOCR](https://rapidai.github.io/RapidOCRDocs/main/en/install_usage/rapidocr/usage/),
[Deepgram pre-recorded audio](https://developers.deepgram.com/reference/speech-to-text/listen-pre-recorded),
[Google Vision OCR](https://docs.cloud.google.com/vision/docs/ocr),
[Cohere joint inputs](https://docs.cohere.com/v2/reference/embed), and
[Qwen mixed inputs](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B).
The optional faster-whisper runtime uses MIT, RapidOCR uses Apache-2.0, and ONNX
Runtime uses MIT. No upstream code, model weights or datasets are bundled here;
model licenses remain properties of the separately provisioned checkpoints.

### Persistent asynchronous tasks

```python
from vane.ai import VideoJobStore

store = VideoJobStore("state/video-jobs.sqlite")
with VideoEmbeddingPipeline(
    profiles=embedding_profiles, source=cloud_source, fusion=cloud_fusion,
    job_store=store, pipeline_version="video-search-v1",
) as pipeline:
    task = pipeline.create(request)  # request["async"] = True
    print(task.id)
    result = task.result()

# After restarting, recreate the pipeline with the same version/configuration:
# task = pipeline.get_task(saved_id)
# if task.status == "interrupted":
#     task = pipeline.resume_task(saved_id)
# Results can also be read without constructing models or a pipeline:
# store.get_task(saved_id, pipeline_version="video-search-v1").result()
# store.list_tasks("video-search-v1")
```

The SQLite store persists JSON requests, execution options, status, sanitized errors
and results (64 MiB maximum per result by default). It never persists executable
Python plans, callbacks, model objects or credentials. Keep the database on a local
filesystem with SQLite locking support. It may contain source URIs and extracted
text; newly created database files have owner-only permissions. This is an embedded
Python task store, without an HTTP server or external job scheduler.

`pipeline_version` is an application-controlled immutable identity for the entire
configuration: sources, extraction/embedding/fusion profiles, model revisions and
preprocessing. Supply that same configuration when resuming; change the version when
changing any of it. Reopening does not automatically submit jobs. Pending unclaimed
and interrupted asynchronous jobs can be resumed explicitly with the same task ID.
Each worker heartbeats an ownership lease (30 seconds by default); an expired job
becomes `interrupted`, and stale workers cannot publish results after a new claim.
Resume reexecutes the read-only embedding plan from its input, not a per-clip checkpoint.
Failed, cancelled and completed jobs cannot be resumed. A caller wait timeout does
not cancel; an explicit cancellation is persisted and observed by the executing worker.

The store manages `create()` requests. Lazy `ingest()`/`batch_ingest()` relations and
their final Lance writes still execute in the caller's connection. No automatic
replay of a Lance append is introduced, including after an uncertain commit outcome.

### Single and batch ingestion into one Lance table

`VideoEmbeddingPipeline.ingest(asset, connection=con, **options)` converts a
single asset dictionary into one input row and calls `batch_ingest()`.
`batch_ingest(assets, **options)` accepts a relation of any number of assets.
Both return the same lazy relation and use the same output schema and final
sink. `create()` also runs this plan over one asset and only formats its output
into `data`/`clips`; it does not run a separate local media loop.

The batch input requires `VARCHAR uri`; optional `VARCHAR asset_id` and
`source_version` columns may be omitted or NULL. Supplied identities must be nonempty.
Missing asset IDs become `uri-sha256:<SHA256 of exact URI UTF-8 bytes>`. Missing
versions are computed by streaming the source bytes as `sha256:<content digest>`
inside the shared probe stage, bounded by `max_input_bytes`. This adds one complete
source read without retaining the video in memory. Custom sources must implement
`source_version(uri)` or receive an explicit version. Use stable URIs, and keep
source bytes immutable for the entire execution/resumption. A version hash cannot
pin a mutable remote object by itself. Explicit IDs/versions avoid the extra read.

Stable
`clip_id` values include the asset identity, version, segmentation size and clip
boundaries. All three entry points resolve identities through the same stage;
a URI is not treated as an immutable media version. Duplicate identities in one
input are rejected. Executing the same input again does not provide upsert semantics.

```python
with vane.connect() as con, VideoEmbeddingPipeline(
    profiles={
        "visual": visual_profile,
        "subtitles": dialogue_profile,
        "transcription": dialogue_profile,
    },
    source=source_with_asr,
) as pipeline:
    result = pipeline.batch_ingest(
        con.read_parquet("assets.parquet"),
        modalities=["visual", "subtitles", "transcription"],
        selection={
            "policy": "all",
            "groups": [{
                "modalities": ["subtitles", "transcription"],
                "policy": "first_available",
                "priority": ["subtitles", "transcription"],
            }],
        },
        clip_seconds=10,
        output="separate",
        resources={
            "probe": {"execution_backend": "ray_task", "cpus": 1},
            "preprocess": {"execution_backend": "ray_task", "cpus": 1, "batch_size": 1},
            "embed": {
                "execution_backend": "ray_actor", "actor_number": 2,
                "cpus": 1, "gpus": 1, "batch_size": 4,
            },
        },
        max_batch_bytes=256 * 1024**2,
    )
    # Attach the installed Lance provider and create a table matching
    # result.columns/result.types before submitting the ingestion query.
    result.insert_into("lance_ns.main.video_clips")

    # The single-asset path has the same options, schema and sink.
    single = pipeline.ingest(
        {"asset_id": "next-video", "source_version": "v1", "uri": "s3://bucket/next.mp4"},
        connection=con,
        modalities=["visual", "subtitles", "transcription"],
        selection={
            "policy": "all",
            "groups": [{
                "modalities": ["subtitles", "transcription"],
                "policy": "first_available",
            }],
        },
        resources={"embed": {"gpus": 1}},
    )
    single.insert_into("lance_ns.main.video_clips")
```

Profiles and the source use the contracts described above. This example requests
GPUs for a GPU visual encoder. CPU models and remote API clients can use
`gpus=0`, including in actors. Configure model devices consistently with the
stage resources. A source passed to any entry point must be serializable
and must not capture a live Vane connection. FILE access and credentials use the
normal Vane worker session configuration.

The execution uses existing operators:

```text
asset scan
  → map_batches: probe once, segment, expand (clip_id, channel group) rows
  → map_batches: preprocess each group
  → map_batches: encode selected content
  → GROUP BY clip_id: assemble typed vector columns
  → optional map_batches: joint-model fusion
  → one Lance INSERT stage
```

Visual, audio and OCR can each be a separate work row. A subtitles → ASR priority
group is one row and evaluates its members in order. Vane schedules batches of
these rows concurrently when resources are available; `batch_size=1` allows even
one clip's independent groups to occupy separate tasks. Larger batches trade
that granularity for lower dispatch overhead. Processing within one batch is
serial. The builder splits probe output into preprocess-sized blocks and sets
the preprocess task minimum to that size, preserving those work boundaries.
No branch operator, branch query or branch checkpoint file is introduced.
Grouping uses the normal distributed aggregation and exchange stages. It is a
blocking aggregation, not a guarantee of immediate per-clip streaming output.

Resource configuration is per stage (`probe`, `preprocess`, `embed`, `fusion`),
shared across every channel handled by that stage. Each accepts
`execution_backend`, `cpus`, `gpus`, `batch_size`, `memory_bytes`, and
`actor_number` for actors only. Probe and preprocess default to CPU `ray_task`,
batch size one. Encode defaults to one CPU `ray_actor`, batch size eight; fusion
defaults to one CPU actor, batch size one. Task and actor selection is independent
of GPU allocation. Tasks construct and close their source/model runtime per
invocation. Actors lazily cache channel encoders across batches; a single actor
may retain several models, whose combined memory must fit its resources.
Pools start incrementally and use the runner's normal prefetch policy; queued
work on the first ready actor may initially execute serially. For workloads
requiring concurrent starts across actors, configure the existing
`VANE_RAY_ACTOR_PREFETCH_DEPTH=1` before creating the Ray runner.
Separate resource pools per channel are outside this interface.

`max_clips` bounds segmentation per asset. `max_batch_bytes` bounds each returned
Arrow stage batch and configures Vane UDF input/output batch budgets. It does not
bound total query memory, decoder temporaries or model weights. Configure source
frame/audio limits and stage batch sizes accordingly. Distributed aggregation
uses the runner's normal resource management rather than a Python coordinator
holding all video results.

Every group returns a row, including normal absence. Assembly rejects duplicate
clip identities or missing/duplicate groups. The output retains `clip_id`,
`asset_id`, `source_version`, `uri`, `start_sec` and `end_sec`. Each requested
channel has its own `<channel>_vector FLOAT[N]` for `separate`/`both` and a
`<channel>_metadata VARCHAR` JSON value containing selection status, reason,
actual channel, times and encoding profile/provider/model. Unselected channel
vectors are NULL. Different dimensions remain distinct typed columns. A priority
group keeps separate channel columns, so two encoding profiles are never treated
as interchangeable. Detailed timelines are controlled by `return_timeline`.

`fused`/`both` carry selected timed content through aggregation to the explicitly
configured joint model. The final relation adds `fused_vector` and
`fused_metadata`; intermediate content BLOBs are removed. No independent vectors
are added together. Decoder/model failures and unsupported fusion combinations
fail the query, with no automatic channel substitution or backend fallback.

`insert_into()` submits one query with one distributed Lance write stage. The
installed Lance provider may use several workers to prepare uncommitted
fragments; its existing coordinator publishes the selected fragments together.
Create the destination with exactly the relation's column order and types.
This API does not add a second commit layer, merge/upsert or a durable job service.
A failed INSERT follows the provider's abort/commit contract. Other sinks such as
`write_parquet()` execute the same lazy plan for local validation.

The focused integration tests in
[`tests/fast/test_ai_video_ingestion.py`](tests/fast/test_ai_video_ingestion.py)
cover independent preprocessing, task/actor stage choices, priority selection,
NULL vectors, distinct vector dimensions, identical results through all three
entry points, synchronous budgets, active-query cancellation, and single/batch
inserts into the same Lance table with one commit per query. The Lance
case requires a matching installed provider; Parquet verification alone does
not establish the Lance commit behavior.
