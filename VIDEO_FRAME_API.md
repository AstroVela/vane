# Streaming video frames

`vane.read_video_frames` and SQL `read_video_frames` emit one row per selected
frame. The `data` column contains decoded `IMAGE('RGB', H, W)` values. Python
fetches materialize them as `vane.Image`; Arrow uses the IMAGE struct storage.
No image extension is required for the video extension to produce IMAGE.

Install `vane-ai[video]` for the Python backend and its decoder and memory
admission dependencies. The native backend uses the optional video extension.

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
native FFmpeg C++ scan, install and load the matching video extension, then set
the backend before constructing and executing the query:

```python
vane.load_installed_extension("video", connection=con)
con.execute("SET video_backend = 'native'")
frames = vane.read_video_frames("clip.mp4", 224, 224, connection=con)
```

Native execution calls the video extension directly. An unavailable native
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
Native execution captures the stream origin when opening the input, using zero
when it is unknown. Estimates discovered while reading packets do not shift
that origin. Index construction, scalar/list expressions and streaming sources
use this same rule.
Without an explicit index, exact frame indices use sequential decoding from
stream start, including for late time windows. The native backend also accepts
reusable indexes for keyframe seeking, as described below. There is no
temporary-file materialization fallback. Python and native codecs do not
promise identical pixels or container-specific metadata.

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
backends allocate pixel payloads for actual rows, including short final batches;
empty scans do not reserve pixel arrays for a vector's unused rows.

The payload budget is not a total-process RSS limit. Codec reference frames,
conversion buffers, temporary Arrow copies, and downstream query operators
consume additional memory. The scan streams bounded batches, while collecting
all rows into a Python list or sorting the complete result adds its own memory
cost. Cancellation is cooperative around I/O and decoding. I/O, permission,
resource, allocation, and cancellation failures propagate under both error
policies.

## Fixed-shape IMAGE types

Combining different IMAGE layouts in `CASE`, `VALUES`, `UNION`, `COALESCE`, or
list construction yields generic IMAGE and preserves each value's pixels.
This also applies to IMAGE leaves inside lists, arrays, maps, and structs.
Equal fixed layouts retain their constraint. Widening to generic IMAGE is
implicit; narrowing to a fixed layout requires an explicit validated cast.

`vane.image_type('RGB', H, W)` and `IMAGE('RGB', H, W)` constrain the existing
IMAGE logical type without changing its physical storage. Modes L, LA, RGB,
and RGBA are supported. Mode, height, and width must be specified together.
`image_type()`/`IMAGE` remains an unconstrained decoded image type.

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
| `video_frames(file, ...)` | LIST of frame records: `file: VIDEOFILE`, the eight temporal fields in the streaming schema, and `data: IMAGE` |
| `video_keyframes(file, ...)` | LIST of RGB IMAGE values selected with the decoder's keyframe flag |
| `get_video_frame_by_idx(file, idx, ...)` | One RGB IMAGE at the zero-based presentation-order decoded index |

Frame records retain the complete source FILE view. For image-only keyframe
and index results, retain the input FILE column alongside the result when
source association is needed. The scalar images use generic IMAGE types;
their values carry the actual dimensions, including after an explicit resize.

Both list functions accept `start_time=0`, `end_time=None`, `width=None`,
`height=None`, and `sample_interval_seconds=None`. `video_frames` additionally
accepts `is_key_frame=None`. Width and height must be supplied together; when
omitted, each selected frame keeps its decoded dimensions. Time selection,
sampling, and exact indices follow the streaming contracts above. Frame-index
lookup closes the decoder after returning the requested frame. All three
functions accept `index=None`; a supplied BLOB selects indexed native access.

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
returns an empty list. NULL index or required scalar options return NULL;
NULL end time, dimensions, keyframe filter, or interval mean no restriction.
Missing frame indices raise an index-range error, or return NULL under the
explicit null policy. Negative indices and invalid options always raise.
Only encoded-format errors may become NULL under `on_error='null'`; I/O,
dependencies, resource limits, cancellation, and unexpected system failures
remain errors. A format error after partial decoding nulls the complete row.

The explicit `video_backend` option applies to all three functions. Python
execution enters PyAV through a C++ scalar bridge using the executing query's
FILE context. Native execution calls the loaded video extension's FFmpeg C++
operators. Binding and expression construction do not open files. Worker
credentials resolve against each original FILE URL and logical byte window;
the Python execution token expires when the scalar row finishes and is never
serialized. No automatic backend or materialization fallback is provided.

## Reusable native video indexes

With the `video` extension loaded and `video_backend='native'`,
`build_video_index(file)` returns an opaque BLOB recording the actual
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
sequential path, which remains available with either backend. Python value
iterators and the Python backend do not consume native indexes.

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
Index format, identity, integrity, seek-reproduction and resource failures are
not suppressed by `on_error='null'` or `'skip'`. Ordinary encoded-format failures
retain the existing error policy. There is no automatic materialization.

`video_index_info(index)` inspects a BLOB without opening a FILE and returns
`frame_count`, `keyframe_count`, `source_bytes`, `index_bytes`,
`build_bytes_read`, and `codec_version`. `vane.video_index_info` constructs its
expression; the `video` extension supplies its implementation.

`video_scan_stats(file, ..., index=None, idx=None)` runs native selection and
returns `bytes_read`, `decoded_frames`, `seeks`, and `selected_frames`. It accepts
the same time/keyframe/sampling options; a non-NULL `idx` selects a single exact
frame instead. It is a separate diagnostic execution which omits output pixel
conversion, not a counter for a preceding query. Bytes include verification
block reads; frames count decoder outputs including discarded preroll, excluding
codec-internal lookahead. Both sequential and indexed measurements use the same
cursor as the native expressions and streaming source. The
[benchmark guide](benchmarking/video_seek/README.md) separates index construction
from repeated query latency and includes loopback HTTP response-byte accounting.
