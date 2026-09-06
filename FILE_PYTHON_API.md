# File Python values and media helpers

`vane.File`, `ImageFile`, `AudioFile`, and `VideoFile` are immutable references
with five public data fields: `url`, `content_type`, `position`, `size`, and
`checksum`. Construction and conversion preserve those fields without I/O.
`position` and `size` must both be absent or both be nonnegative integers whose
sum fits in a signed 64-bit value. Files contain no resolved credentials.

See the [FILE SQL contract](external/duckdb/extension/file/README.md) for
metadata, byte windows, comparison, identity, and the current distributed I/O
boundary.

## Media classification and conversion

Every File value, including the three media subclasses, provides:

```python
file.is_image()  # bool
file.is_audio()  # bool
file.is_video()  # bool
file.as_image()  # ImageFile
file.as_audio()  # AudioFile
file.as_video()  # VideoFile
```

Classification uses the declared logical subtype first. For a generic File,
it uses `content_type` when present, or a recognized URL suffix when absent.
MIME family matching ignores case, surrounding whitespace, and parameters.
An unrecognized or empty hint returns `False`. These calls do not open the
URL or verify content; a URL suffix or MIME declaration is only a routing hint.

Conversion declares the target subtype and returns a new value with exactly
the source's five fields. A generic File can become any of the three media
subtypes. An already matching subtype is accepted; conversion between different
media subtypes raises `TypeError`. Decoders validate the encoded content and
its MIME declaration when executed.

```python
import vane

source = vane.File("s3://example-bucket/photo.png", "image/png", 128, 4096)
image = source.as_image()
assert image.is_image()
assert image.position == source.position
assert image.size == source.size
assert vane.file_type(vane.MediaType.image()).is_file()
```

The schema specialization survives SQL parameters, persistence, and governed
UDF input/output materialization under the existing FILE contract. Python
equality compares the complete immutable value and subtype. SQL expression
equality requires the same FILE subtype and retains SQL NULL semantics.

## Metadata values

`file.stat(*, connection=None)` returns a frozen, pickleable `vane.FileStat`:

| Attribute | Python type | Meaning |
| --- | --- | --- |
| `url` | `str` | Backing-object URL |
| `object_size` | `int \| None` | Whole-object size, including for a ranged File |
| `last_modified` | `datetime \| None` | Connector modification timestamp |
| `version` | `str \| None` | Connector object version |
| `etag` | `str \| None` | Connector ETag |
| `content_type` | `str \| None` | Available metadata MIME hint |

Use attributes such as `stat.object_size`. `dataclasses.asdict(stat)` produces
a dictionary when needed. Metadata is a snapshot; unavailable fields are
`None`, and a later object change does not modify an existing FileStat. ETags
are not promoted to checksums. SQL and expression `file_stat` continue to
return the documented STRUCT.

`file.exists(*, connection=None)` returns `True` for an accessible logical
view and `False` for a missing object, non-file object, or out-of-bounds range.
It raises `vane.IOException` if access cannot be determined because of a
permission, network, configuration, or connector failure. Other system errors
propagate. SQL and expression `file_exists` represent indeterminate access as
NULL. The explicit `connection` selects the execution context for both value
methods.

## Expression options

The function and Expression-method forms share defaults, validation, and
resource options:

| Function / method | Options |
| --- | --- |
| `file_mime_type` | `detect` |
| `image_file_metadata` | `max_bytes`, `max_pixels` |
| `decode_image_file` | `mode`, `on_error`, `max_input_bytes`, `max_pixels`, `max_decoded_bytes` |
| `audio_metadata` | `max_bytes` |
| `video_metadata` | `max_bytes` |

These arguments accept the same literals or Expressions as their function
forms. Resource options are keyword-only. For example:

```python
expression = vane.col("image_file")
a = vane.decode_image_file(expression, "RGB", max_pixels=1_000_000)
b = expression.decode_image_file("RGB", max_pixels=1_000_000)
```

Both forms construct the same media expression. Codec execution occurs when
the query runs, using the connection's selected Python or native backend.
See [native extensions](NATIVE_MEDIA_EXTENSIONS.md) for installation, loading,
and backend selection, and [video frame APIs](VIDEO_FRAME_API.md) for the
shared frame-selection options. An explicit native request never falls back to
Python media execution.

## Video value metadata buffering

```python
metadata = video_file.metadata(
    buffer_size=64 * 1024,
    max_bytes=8 * 1024 * 1024,
    connection=connection,
)
```

`buffer_size` is an optional positional or keyword argument. It controls the
PyAV I/O buffer and bounded read-ahead cache for this Python value method.
`max_bytes` is the independent total source-read budget; a larger buffer cannot
increase it. The I/O buffer and read-ahead are capped by that budget. A
decoder request larger than the buffer can still be served with one bounded
source read. All accesses stay inside the File's logical byte window.

The buffer must be a positive integer fitting in a C `int`. Invalid options
fail before importing the codec or opening the file. Metadata probing retains
its existing fetch-count and timeout limits and never decodes pixel frames.
Image and audio metadata retain their existing fields, including integer audio
frame counts and duration, and nullable unknown video metadata.
