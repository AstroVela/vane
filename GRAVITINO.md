# Gravitino catalog and Filesets

The optional `gravitino` C++ extension connects Vane to Apache Gravitino 1.3
FILESET catalogs. It uses DuckDB's `StorageExtension`, `Catalog`, and
`FileSystem` interfaces. Python's `vane.catalog.GravitinoCatalog` is a wrapper
around those native operations; it does not read data through a Python SDK.

For the implementation plan and later metadata discovery and Iceberg work, see
the [Gravitino design and roadmap](GRAVITINO_DESIGN.md). The capabilities below
describe the current implementation, including Fileset batch writes.

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
These source changes do not publish a provider wheel.

Batch writes add `FileSystem::ResolveWriteTarget`, which changes the native
filesystem ABI. Rebuild the base and all optional native providers against the
same engine sources. A matching Python package version alone does not establish
native compatibility; retain the engine and artifact identity checks described
in the development workflow.

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
On local connections, prepared metadata queries and Fileset scans resolve
fresh remote metadata on every execution. Fileset scans also refresh glob
matches; they do not reuse locations or file lists captured by an earlier
`PREPARE` or `EXECUTE`.

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

## Fileset batch writes

`GravitinoCatalog.write_files` writes Parquet or CSV through native COPY and
returns a frozen `FilesetWriteResult` containing the exact output files, row
count, byte count, selected location, operation ID and physical output base.
Storage access uses the connection's existing settings. Supported combinations
are:

| Runner | Absolute local / `file:` locations | Shared S3 locations |
| --- | --- | --- |
| `local` (`local-fast` alias) | Supported | Supported |
| `ray` | Rejected before writing | Supported through the existing FTE COPY protocol |

Use a relation belonging to the catalog's connection. For local files, set
`VANE_RUNNER=local` before creating that connection:

```python
result = catalog.write_files(
    connection.sql("SELECT i, 'example' AS label FROM range(10) AS t(i)"),
    "videos",
    "training",
    "exports",
    format="parquet",
    compression="zstd",
)
rows = connection.read_parquet(list(result.files)).order("i").fetchall()
```

Each invocation creates a distinct batch directory below the requested relative
prefix, including when the prefix is empty. It does not replace previous output
or deduplicate repeated calls. CSV uses UTF-8, commas and a header. Supply
`partition_by=["column_name", ...]` for native Hive-style partition directories.
Partition columns remain in every file. Empty input produces a zero-row output
at the batch root, including with partitioning; CSV preserves column names,
while Parquet also preserves types. When reading partitioned files, use
`hive_partitioning=False` to preserve the stored partition values, including a
literal `__HIVE_DEFAULT_PARTITION__` string. CSV empty-input types must be
supplied to the reader explicitly when needed.

The attachment must be writable and the connection must be in auto-commit mode.
Output paths cannot contain glob patterns, dot segments, backslashes or percent
escapes. Location selection follows the attachment and Fileset rules below.
The physical target is resolved once per invocation. Output files and statistics
come from that COPY execution or its committed manifest, not a later directory
scan. Ray workers receive the frozen physical destination and storage settings,
without the Gravitino attachment or bearer token. They do not resolve the
Fileset again during task retries.

There is no local batch manifest or directory-level atomic publication;
`run_id` and `manifest_path` are `None`. Readers that list the directory can see
partial failed output. Read `result.files` to select a successful batch. Failures
can leave files belonging to that invocation. A `CopyResultUnavailableError`
means COPY completed but its result could not be returned; it has
`safe_to_retry=False` and includes the operation ID and output base. Do not rerun
the write automatically in that case.

For Ray S3 writes, `run_id` and `manifest_path` identify the durable commit.
`files` contains only attempts selected by that commit. `cleanup_state` is
`"complete"` or `"pending"`; `cleanup_warnings` retains the runner's diagnostics.
After worker loss, the data can commit while teardown remains pending. Closing
the connection can then report a cleanup error; retain the committed receipt
and resolve cleanup separately instead of repeating the write. A local
`"complete"` state does not add a durable manifest or atomic publication.
Cleanup warnings do not turn a committed write into a failed write. If commit-marker publication
or observation is inconclusive, the existing `CopyOutcomeUnknownError` retains
the available recovery identifiers and prohibits automatic resubmission.
For an error that includes `base_path` and `run_id`, inspect the original run
using the same storage-configured connection:

```python
from vane.runners.ray.committed_copy import inspect_copy_direct_write_run

status = inspect_copy_direct_write_run(error.base_path, error.run_id, conn=connection)
if status["state"] == "COMMITTED":
    files = [entry["final_path"] for entry in status["files"]]
```

Inspection is read-only. `UNCOMMITTED` does not prove that an earlier publication
cannot complete. Stop the original write and resolve its state before deciding
whether to retry. Explicit force-abort is an operator decision that can discard
a run; it is not an automatic recovery step.

Cancellation is reconciled with commit. An interruption observed before the
terminal outcome can produce an interruption/unknown-outcome error, or return
the committed receipt if publication won the race. Task retry requires live
execution capacity; loss of every Ray worker is not an automatic cluster
recovery guarantee.

`result.files` contains literal physical filenames. Some format readers expand
glob syntax in their string inputs. For registered roots containing glob
metacharacters, use a literal file-opening API such as
`vane.open_file(path, "rb", connection=connection)` with your format reader.
Passing those names to a glob-expanding reader can select a different file set.

Native SQL COPY and the existing Parquet writer also accept GVFS directory
destinations with `APPEND` plus `PER_THREAD_OUTPUT` or nonempty `PARTITION_BY`.
Use `WRITE_PARTITION_COLUMNS` to preserve a readable empty partitioned output.
Fixed-file output, overwrite, custom filenames and temporary-file publication are
rejected before writing. Prepared local COPY statements resolve fresh
Fileset metadata at every execution; EXPLAIN and PREPARE create no output.
The existing Python writer return types are unchanged.

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
- Supported content locations are absolute local paths, local `file:` paths,
  and S3 (`s3a://` normalizes to `s3://`). Unsupported schemes fail explicitly.
  Locations use Hadoop `Path.toString()` semantics: `%20` is a literal directory
  name, not an encoded space; local `#` and `?` characters are also literal.
  The registered root is never a glob pattern. Only the relative argument to
  `files()` or `gvfs` supports glob syntax. Globbing a root with backslashes
  and S3 roots containing `?` or backslashes are unsupported.
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
  The first path component is the local attachment alias. Raw filesystem write
  handles remain unsupported; COPY resolves validated destinations to physical
  paths before selecting its storage strategy.
- Configure attachments before enabling `configure_local_runtime()` on the
  local-runtime branch. Runtime query admission does not permit metadata writes;
  setup and management happen outside its read-only query phase.

ATTACH accepts `TOKEN` (optional bearer authentication), `TIMEOUT_MS` (default
30000, maximum 300000), `MAX_RESPONSE_BYTES` (default 2 MiB, maximum 16 MiB),
`LOCATION_NAME`, and the standard `READ_ONLY` option. HTTP requests have an
overall transport timeout, capped response bodies and headers, no redirects,
and no compressed response decoding. Discovery is capped at 4096 identifiers.

Relational table scans/writes and remote catalog creation/deletion
are outside this Fileset integration. Use the appropriate Lance/Iceberg connector
for table operations.

## Validation

Iceberg provider compatibility and native local REST reads have a separate
[M4a qualification record](GRAVITINO_ICEBERG.md). They use an independently
built Iceberg/Avro provider pair and the service's Iceberg REST endpoint; the
Fileset catalog attachment and metadata tests below do not qualify that path.

Tests use a local HTTP fixture implementing the Gravitino 1.3 REST shapes and
actual native file reads. Run non-Ray and Ray tests in separate processes:

```bash
export VANE_TEST_GRAVITINO=provider # use static for a static development build
scripts/run_installed_pytest.sh tests/fast/test_gravitino.py tests/fast/test_gravitino_write.py -m 'not real_ray'
scripts/run_installed_pytest.sh tests/fast/test_gravitino.py tests/fast/test_gravitino_write.py -m 'real_ray and not ray_cluster_owner'
scripts/run_installed_pytest.sh tests/fast/test_gravitino_write.py -m 'real_ray and ray_cluster_owner'
```

CI also runs `tests/fast/test_gravitino_live.py` against the real
`apache/gravitino:1.3.0` Docker image pinned by digest. A separate required job
installs the matching base wheel and signed provider from the native build,
waits for service readiness, then runs local and Ray tests in separate
processes. It checks catalog/schema/Fileset metadata mutations, the server's
rejection of schema comment changes, EXTERNAL file retention, and native file
reads and batch writes through registered local and S3 locations. S3 acceptance
uses the HTTP object-store fixture alongside a real catalog configured with
`disable-filesystem-ops=true`, so Gravitino manages metadata and Vane accesses
storage. See the [Fileset catalog properties](https://gravitino.apache.org/docs/1.3.1/fileset-catalog/).
This does not replace qualification against a production object store. The container is removed on success or
failure, with server logs printed on failure.

To run these acceptance tests against an explicitly provisioned Gravitino 1.3
test service using its default anonymous authentication:

```bash
export VANE_TEST_GRAVITINO=provider
export VANE_TEST_GRAVITINO_URL=http://127.0.0.1:8090
export VANE_TEST_GRAVITINO_SHARED_DIR=/absolute/path/to/test-files
scripts/run_installed_pytest.sh -m 'external_service and not real_ray' tests/fast/test_gravitino_live.py
scripts/run_installed_pytest.sh -m 'external_service and real_ray' tests/fast/test_gravitino_live.py
```

Create the shared directory before starting Docker and bind-mount it at the
same absolute path inside the container. The tests create and remove uniquely
named metalakes and test directories; they use EXTERNAL Filesets. Do not point
them at a production service. Missing explicit environment settings skip these
tests in local runs. CI supplies both settings and the extension provider.

Cloud object stores and deployed authorization policies are outside this CI
acceptance; credential and transport failures remain covered by HTTP fixtures.
