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
Exact frame indices currently require sequential decoding from stream start,
including for late time windows. Indexed seeking is tracked separately in
#714. There is no temporary-file materialization fallback. Python and native
codecs do not promise identical pixels or container-specific metadata.

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

This API is the streaming part of #756. The single-row frame-list functions,
frame-index lookup, and their Expression methods are separate follow-up work.
