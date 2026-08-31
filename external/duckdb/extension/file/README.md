# FILE SQL extension

`FILE` is a governed, read-only reference. The extension never copies resolved
credentials into its five persisted fields. Functions that access storage open
the URL at execution time through the current `ClientContext` file opener, so
connector selection, Secret scope matching, retries, and credential refresh
remain connector-owned.

## Media specialization contract

`IMAGEFILE`, `AUDIOFILE`, and `VIDEOFILE` are schema-level specializations of
the same five-field FILE storage. They add no fields, credentials, or decoded
content. `image_file`, `audio_file`, and `video_file` accept a URL, a generic
FILE, or an already matching specialization. Their default path is pure and
performs no I/O. Passing `TRUE` as the second argument performs bounded
magic-byte inspection through the existing resolver and rejects content that
does not match the declared media family.

Generic metadata, reader, identity, UDF, and AI consumers accept the complete
FILE family. FILE-valued functions such as `file_enrich` preserve the input
specialization. Direct equality and inequality require the exact same alias;
the explicit location and content identity functions can compare different
family members. Media decoding is intentionally outside this extension layer.

## Metadata contract

`file_stat(file)` returns this connector-neutral struct:

```sql
STRUCT(
    url VARCHAR,
    object_size UBIGINT,
    last_modified TIMESTAMP,
    version VARCHAR,
    etag VARCHAR,
    content_type VARCHAR
)
```

- `object_size` is the backing object's size, not a ranged FILE's logical
  size. `file_size(file)` returns the logical size and can therefore use a
  declared range without I/O.
- `last_modified`, `version`, `etag`, and `content_type` are nullable. Values
  exposed by a connector are returned when available and may become stale
  after the call. Local files do not synthesize a provider version. A MIME type
  inferred from a URL suffix is a hint. Content sniffing is requested
  explicitly with `file_mime_type(file, 'content')` or used after metadata by
  `'auto'`. Provider ETags remain ETags; they are never promoted to checksums or
  versions.
- `to_file(path)` opens the object and returns the complete logical view as
  `position = 0, size = object_size`. `try_to_file` returns NULL for storage,
  network, permission, configuration, or missing-connector failures.
- `file_exists` returns false for a missing object, a non-file object, or a
  declared range outside the backing object. It returns NULL when access cannot
  be determined because of permission, network, configuration, or connector
  failures.

`file_enrich(file, fields)` accepts `size`, `content_type`, and `checksum`.
Missing content types use connector metadata, then a recognized URL suffix,
and otherwise sniff exactly the FILE's logical byte window. Missing checksums
are computed as lowercase SHA-256 over that same window. Enrichment never
overwrites an already populated field.

## Identity contract

Identity functions do not perform I/O:

- `file_same_location` compares URL and logical range. Whole-object versus
  explicit-range references are indeterminate (NULL), even when the explicit
  range may cover the whole object.
- `file_locator_id` is
  `file-locator-v1:sha256:<hex>`. The digest covers the byte string
  `vane:file-locator:v1`, a NUL separator, the URL length as an unsigned
  64-bit big-endian integer, the URL bytes, a whole/range marker, and—when
  ranged—the position and size as unsigned 64-bit big-endian integers.
- `file_same_content` first proves inequality from two known different logical
  sizes, treats two known zero-length views as equal, and otherwise compares
  checksum digests only when their case-insensitive algorithm names match.
  Insufficient or incomparable evidence returns NULL.
- `file_content_id` is `file-content-v1:empty` for a known empty view,
  `file-content-v1:checksum:<lowercase-algorithm>:<digest>` when a checksum is
  present, and NULL without content evidence. Digest bytes remain opaque and
  case-sensitive.

## I/O boundary

Every storage-facing function validates the exact FILE alias, non-NULL and
NUL-free URL, paired and non-negative range, range overflow, and checksum token
before opening a connector. Reads prefer the positional FileSystem API and fall
back to seek plus bounded sequential reads for connectors without positional
reads. Both paths reject a short object and never expose bytes outside the
declared `[position, position + size)` window.

Content MIME detection is a bounded, best-effort magic-byte hint, not format
validation. It reads through the 4 KiB boundary plus the eight-byte HDF5
signature first; strong prefix evidence avoids any additional HDF5 I/O. The
built-in rules cover common PNG, JPEG, GIF, WebP, ISO-BMFF, MP3, WAV, Ogg, MPEG,
HDF5, PDF, ZIP, and HTML content. Unknown content returns `NULL`. If the prefix
is inconclusive, HDF5 additionally probes the remaining legal power-of-two
user-block offsets with one eight-byte logical-range read per offset, so
detection remains logarithmic in the FILE view size. Decoders and downstream
providers remain responsible for validating content before consuming it.

Resolver opens carry DuckDB's nonblocking flag so native local special files
can be rejected from the opened handle without a check/open race. Registered
Python filesystems do not yet implement that open contract and fail before
performing I/O; their provider-specific execution support is intentionally
deferred.

## Distributed boundary

Ray execution treats the source Driver, Ray object transport, and the Workers
participating in one Vane query as one trusted runtime boundary. FILE values
remain the canonical five-field value and never carry connector configuration
or credentials. The immutable connection/session snapshot is transported
separately and applied to each Worker-owned DuckDB connection before FILE
functions execute.

The official Vane build loads statically linked `file` and `httpfs` extensions,
and its existing snapshot records and verifies their exact identities before
Worker execution. This makes HTTP and S3 connector lookup available without
runtime extension installation or autoload. Existing Vane session handling
resolves and refreshes supported AWS credentials per connection session.
Explicit supported DuckDB connection settings retain their normal precedence.
DuckDB `CREATE SECRET` objects are not serialized.

This first distributed contract is for controlled, single-trust-domain Ray
deployments. A local path names a Worker-visible path and is portable only when
all participating Workers see the same object at that path. Multi-scope Secret
selection and untrusted-cluster isolation remain a later execution-layer
contract; neither requires changing FILE's persisted representation.
