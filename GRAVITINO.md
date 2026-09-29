# Gravitino catalog and Filesets

The optional `gravitino` C++ extension connects Vane to Apache Gravitino 1.3
FILESET catalogs. It uses DuckDB's `StorageExtension`, `Catalog`, and
`FileSystem` interfaces. Python's `vane.catalog.GravitinoCatalog` is a wrapper
around those native operations; it does not read data through a Python SDK.

## Build and load

Build a matching loadable artifact using the [development workflow](DEVELOPMENT.md):

```bash
export SKBUILD_BUILD_DIR="$PWD/build/python-release"
export SKBUILD_CMAKE_BUILD_TYPE=Release
uv pip install . --no-build-isolation '-Ccmake.define.VANE_LOADABLE_EXTENSIONS=gravitino'
cmake --build "$SKBUILD_BUILD_DIR" --target vane_loadable_extensions
```

Package and sign the artifact using the existing
[extension-wheel workflow](DEVELOPMENT.md#building-an-optional-extension-wheel),
including `LICENSES/vcpkg-binary-dependencies.txt` for the linked native
dependencies. Install the matching provider on the coordinator and every Ray
worker, then call `vane.load_installed_extension("gravitino", connection=connection)`.
This PR adds source and packaging support; it does not publish a provider wheel.

For an explicitly static development build, add `gravitino` to `BUILD_EXTENSIONS`
and use `connection.execute("LOAD gravitino")`.

## Register and read resources

Start with an existing Gravitino metalake and FILESET catalog. Attach does not
create a remote catalog. The endpoint is the server base URL, optionally with a
deployment prefix; the extension appends `/api`.

```python
import os
import vane
from vane.catalog import GravitinoCatalog

connection = vane.connect()
vane.load_installed_extension("gravitino", connection=connection)
catalog = GravitinoCatalog.attach(
    connection,
    "media",
    endpoint=os.environ["GRAVITINO_URL"],
    metalake="production",
    catalog="media_files",
    token=os.environ["GRAVITINO_TOKEN"],
)
catalog.create_schema("videos")
catalog.create_fileset(
    "videos", "training",
    storage_location="s3://example-bucket/training/",
    properties={"purpose": "agent-context"},
)
files = catalog.files("videos", "training", "*.mp4")
files.project("url, object_size, file").show()
catalog.alter_fileset("videos", "training", [
    {"@type": "setProperty", "property": "owner", "value": "context"},
])
```

The equivalent native entry points include:

```sql
ATTACH 'media_files' AS media
    (TYPE gravitino, ENDPOINT 'http://localhost:8090', METALAKE 'production');
CREATE SCHEMA media.videos;
SELECT * FROM gravitino_schemas('media');
SELECT * FROM gravitino_filesets('media', 'videos');
SELECT * FROM gravitino_fileset('media', 'videos', 'training');
SELECT * FROM gravitino_files('media', 'videos', 'training', '*.mp4');
PRAGMA gravitino_alter_fileset(
    'media', 'videos', 'training',
    '{"updates":[{"@type":"updateComment","newComment":"Training videos"}]}'
);
```

`gravitino_catalog`, `gravitino_schema`, and `gravitino_fileset` return `name`
and a JSON string in `metadata`. The plural discovery functions return `name`
with NULL `metadata`; loading details is explicit. `gravitino_files` returns
the same columns as `list_files`, including a typed `FILE` column. Its
`recursive` named argument defaults to false.

Metadata mutation pragmas are `gravitino_create_fileset(alias, schema, json)`,
`gravitino_alter_fileset(alias, schema, name, json)`,
`gravitino_drop_fileset(alias, schema, name)`,
`gravitino_alter_schema(alias, schema, json)`, and
`gravitino_alter_catalog(alias, json)`. Create JSON uses Gravitino's `name`,
`type`, `properties`, optional `comment`, and `storageLocation` or
`storageLocations` fields. Alter JSON uses an `updates` array. Supported changes
are `setProperty` and `removeProperty` for all three resources,
`updateComment` for catalogs and Filesets, and `rename` for Filesets.
Gravitino 1.3 does not support changing schema comments.
Schema creation/deletion use SQL `CREATE SCHEMA` and `DROP SCHEMA`.

## Execution and storage contracts

- Local execution uses the connection's native Catalog and FileSystem directly:
  bind the Fileset, resolve its location, then execute the native file scan in
  the same process. It does not require Ray, a driver, or workers. Distributed
  scan callbacks are used only when the caller selects the Ray runner.
- Metadata writes run on the connection, require auto-commit, and are rejected
  for `READ_ONLY` attachments. They do not provide rollback or cross-resource
  transactions. Transport failures after a write may have an unknown outcome;
  the extension never automatically retries writes.
- Gravitino's MANAGED/EXTERNAL semantics apply. Dropping MANAGED Filesets or
  cascading their schema can delete physical files through Gravitino. EXTERNAL
  Fileset deletion retains its files. `create_fileset` defaults to EXTERNAL.
- Supported content locations are absolute local paths, local `file:` URIs,
  and S3 (`s3a://` normalizes to `s3://`). Unsupported schemes fail explicitly.
  Storage credentials are configured using the existing Vane storage settings;
  the Gravitino bearer token is only for metadata requests.
  The Python wrapper binds the token as a SQL parameter, and the extension
  removes it from the public attachment options. Native SQL callers should
  likewise bind `TOKEN $token` to keep credentials out of query logs.
  Direct Python file opens must use the configured connection. Python UDFs that
  open files on their own connection must configure that connection's storage
  access; returning a FILE value does not transfer connection credentials.
- Named locations use the attachment's `LOCATION_NAME`, otherwise the Fileset's
  `default-location-name` property, otherwise Gravitino's reserved unnamed
  location `unknown`. Missing selections fail; no other location is chosen.
- Relative content paths cannot contain dot segments, backslashes, or percent
  escapes. Filesets provide location discovery, not an operating-system sandbox.
- `gravitino_files` resolves the physical path while binding on the querying
  connection, before using Vane's existing file scan. When using Ray, workers
  must access the same storage. Use this
  entry point for distributed reads; an unresolved `gvfs://` FILE literal does
  not carry a catalog attachment to a worker.
  Bound Gravitino plans contain resolved metadata or physical paths, so the
  connection snapshot does not replay the attachment or transport its token.
- Native `gvfs://fileset/<attachment>/<schema>/<fileset>/<path>` opens work on
  the attached connection, including `vane.open_file(..., connection=...)`.
  The first path component is the local attachment alias. Content writes through
  this filesystem are unsupported.
- Configure attachments before enabling `configure_local_runtime()` on the
  local-runtime branch. Runtime query admission does not permit metadata writes;
  setup and management happen outside its read-only query phase.

ATTACH accepts `TOKEN` (optional bearer authentication), `TIMEOUT_MS` (default
30000, maximum 300000), `MAX_RESPONSE_BYTES` (default 2 MiB, maximum 16 MiB),
`LOCATION_NAME`, and the standard `READ_ONLY` option. HTTP requests have an
overall transport timeout, capped response bodies and headers, no redirects,
and no compressed response decoding. Discovery is capped at 4096 identifiers.

Relational table scans/writes, remote catalog creation/deletion, and file uploads
are outside this Fileset integration. Use the appropriate Lance/Iceberg connector
for table operations.

## Validation

Tests use a local HTTP fixture implementing the Gravitino 1.3 REST shapes and
actual native file reads. Run non-Ray and Ray tests in separate processes:

```bash
export VANE_TEST_GRAVITINO=provider # use static for a static development build
scripts/run_installed_pytest.sh tests/fast/test_gravitino.py -m 'not real_ray'
scripts/run_installed_pytest.sh tests/fast/test_gravitino.py -m real_ray
```

These tests do not constitute acceptance against a deployed Gravitino server,
object store, or its authorization policies.
