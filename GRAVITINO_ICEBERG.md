# Gravitino Iceberg provider qualification

M4a tracks an independently built Iceberg provider and native local reads
through Gravitino's Iceberg REST service. The tracking issue is
[#1064](https://github.com/AstroVela/vane/issues/1064); production wheel
publication remains [#619](https://github.com/AstroVela/vane/issues/619).
M4a established the provider qualification. M4b adds the local Python interface
below, tracked in [#1066](https://github.com/AstroVela/vane/issues/1066).
M3's general Iceberg capability flags remain conservative; this record covers
the explicitly qualified local provider, service and feature combinations.

## Fixed inputs

| Input | Revision/version |
| --- | --- |
| Vane base | `f54912b6993d8c19c65e2d26e3965e5e24dcfb2e` |
| DuckDB official release line | `v1.5.5` |
| Engine tree SourceID | `830d7bf6485ea8da211a2342a4b3f86684054db5` |
| Engine fork revision | `493f565f68d73a86c089bf758708d14dbce86d6e` |
| Iceberg source | `duckdb/duckdb-iceberg@45163a28e0ed6a2071a82a1bf1dd432d0216cf9c` plus Vane's compatibility patch |
| Avro provider source | `duckdb/duckdb-avro@f9d590297485f0318f480372c70bdd852826e258` |
| Native SDK | [separate profile](packages/vane-iceberg-provider/vcpkg.json), builtin baseline `44819aa2a6c10e56065e2b0330e7d6c89d1d2574` |
| Avro C registry | `duckdb/vcpkg-duckdb-ports@f13de79b66345fe31d3f9c119517613023d194a8` |
| Avro C source | `duckdb/duckdb-avro-c@8af400279c445a81b8552a7670d8c1ebd92ba34a` |
| Real service | `apache/gravitino:1.3.0@sha256:04c74cee17946865fa20186c032fda2ed7d1722468e2d58aa32114ea3b35b057` |
| Independent fixture client | PyIceberg 0.12.0, pyiceberg-core 0.10.1, PyArrow 25.0.0 |

The extension source revisions are already pinned in the engine's extension
configuration. Vane carries its own compatibility patch; qualification and
later distributed support do not depend on an upstream merge.
The patch SHA-256 is
`5c4b4911c6a4024b407f0b7d39ad0b261fc9f63298f0d803d5a69248bd199f2b`.

## Qualification scope

The fixture uses the real Gravitino 1.3.0 auxiliary Iceberg REST service with
its memory catalog backend and a test-owned local warehouse. The server and
client share the same absolute filesystem path. PyIceberg independently creates
format-v2 tables and commits two Parquet batches totalling 4101 rows. Values
include NULL and Unicode strings. A valid uncommitted Parquet file is placed
beside the committed files, so matching a directory glob cannot pass the data
comparison. A separate empty table checks schema retention.

Vane uses its native local runner, an explicit READ_ONLY Iceberg attachment,
disabled extension downloads/autoloading, and the signed installed provider
resolver. Only the Iceberg provider is requested; its descriptor must load the
exact Avro dependency first. The fixture forbids Ray initialization and uses
an isolated secret directory.

On Linux x86-64, Python 3.12.3 and PyArrow 25.0.0, both native artifacts compile
and load through the signed test-provider resolver. All six real-service
acceptance cases pass. This result qualifies the following narrow matrix:

| Operation or feature | Result |
| --- | --- |
| Source build and signed local provider resolution | Passed; exact Avro dependency loads before Iceberg |
| Existing REST table, format v2, unpartitioned Parquet | Passed; all 4101 independently verified rows match |
| Uncommitted Parquet file beside committed data | Excluded from the Iceberg result |
| Projection, tested numeric filters and aggregation | Passed |
| Empty table | Preserves BIGINT/VARCHAR schema with zero rows |
| READ_ONLY attachment | INSERT rejected and independently observed snapshot unchanged |
| Scan-plan serialization | Explicitly rejected by the unimplemented Iceberg callback |
| Other formats, table features, platforms and runners | Unqualified |

The plan factory exposes the native serialization error through its existing
`ValueError` JSON envelope. A source-level `NotImplementedException` does not
imply that this particular Python factory raises that native exception class.

## Reproduce the provider build

Follow [DEVELOPMENT.md](DEVELOPMENT.md) for the base SDK, noneditable Python
environment, and incremental Release build. Prepare the additional SDK with
the [qualification profile instructions](packages/vane-iceberg-provider/README.md).
Check out the fixed Iceberg and Avro sources separately, then apply
`packages/vane-iceberg-provider/patches/iceberg-vane-local.patch` to Iceberg.

Export these source overrides for **every** configure/build command, including
CMake's automatic regeneration. `CMAKE_PREFIX_PATH` identifies the extra SDK;
the base SDK continues to follow the development guide's `VCPKG_INSTALLED_DIR`
and triplet selection.

```bash
export DUCKDB_ICEBERG_DIRECTORY=/absolute/path/to/patched-iceberg
export DUCKDB_AVRO_DIRECTORY=/absolute/path/to/pinned-avro
export CMAKE_PREFIX_PATH="$PWD/build/iceberg-sdk/x64-linux-release${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
export SKBUILD_BUILD_DIR="$PWD/build/python-release"
export SKBUILD_CMAKE_BUILD_TYPE=Release
uv pip install . --no-build-isolation \
  '-Ccmake.define.VANE_LOADABLE_EXTENSIONS=avro;iceberg' \
  '-Ccmake.define.VANE_ENABLE_TEST_EXTENSION_SIGNING_KEY=ON'
cmake --build "$SKBUILD_BUILD_DIR" --target vane_loadable_extensions
```

The base module keeps Avro/Iceberg unlinked. Both artifacts are self-contained
and must match the installed Vane engine and SDK identities. Follow the
[test signing and provider-wheel workflow](DEVELOPMENT.md#building-an-optional-extension-wheel):
package Avro first, then pass its exact wheel to the Iceberg builder with
`--dependency-wheel` and `--dependency-trust-identity vane-ci-test-key`.
Use `--test-only`, retain both upstream MIT notices and the native dependency
notice bundles, and install both exact output wheels. The profile is a Linux
qualification build, not a portable production wheel release.

## Run the real-service acceptance

Enable `iceberg-rest` in the pinned Gravitino container, publish its REST port
on loopback, and bind-mount a unique temporary directory at the same absolute
path. Configure its warehouse under that directory. The REST URL includes the
`/iceberg` prefix and is separate from Gravitino's `/api` metadata endpoint.
The qualification uses the service's existing anonymous mode.

```bash
uv pip install -r packages/vane-iceberg-provider/requirements-test.txt
export VANE_TEST_GRAVITINO_ICEBERG_URL=http://127.0.0.1:9001/iceberg
export VANE_TEST_GRAVITINO_ICEBERG_SHARED_DIR=/absolute/path/to/unique/test-directory
scripts/run_installed_pytest.sh tests/fast/test_gravitino_iceberg_live.py -m external_service
```

The helper creates a unique namespace and directories, independently verifies
the committed rows, then drops only that namespace and removes its files.
Fixture-only metadata/data directories are writable by both the Java server
and Python client. Missing both environment settings skips the module; partial
configuration or an unavailable opted-in SDK/provider/service fails explicitly.
Remove the task-owned container and temporary mount after testing.

The pinned PyIceberg client is configured with its existing `noop` auth mode
and an explicit URI, independently of home-directory catalog configuration.
This fixture configuration does not add a Vane authentication method.

## Local Python read interface (M4b)

Set `VANE_RUNNER=local` before opening the connection. Load the matching
installed provider explicitly, then attach an existing catalog:

```python
import os

import vane
from vane.catalog import IcebergCatalog

os.environ["VANE_RUNNER"] = "local"
with vane.connect() as connection:
    vane.load_installed_extension("iceberg", connection=connection)
    catalog = IcebergCatalog.attach(
        connection,
        "lake",
        endpoint="http://127.0.0.1:9001/iceberg",
        warehouse="",
    )
    rows = catalog.table("analytics", "events").filter("id >= 100").project("id, label").limit(10).fetchall()
    catalog.detach()
```

`endpoint` is the explicit Iceberg REST HTTP(S) URL, including its service
prefix. Embedded credentials, URL queries/fragments, whitespace and control
characters are rejected before native access. It is separate from Gravitino's
metadata endpoint. `warehouse` is passed as the native ATTACH path and REST
warehouse routing parameter; `""` selects the server default used by this
fixture. Named warehouse/multi-catalog routing remains service-specific and
unqualified here.

Attachments always use `READ_ONLY` and `AUTHORIZATION_TYPE 'none'`. The helper
does not download extensions, create a remote catalog, import storage settings
from metadata, or add authentication. Missing loaded providers fail with a
clear error. General provider publication remains #619.

`table(namespace, name, *, snapshot_id=None)` returns an ordinary native
`DuckDBPyRelation`. Each name is one quoted identifier: `namespace` is the
native provider's schema name, not an arbitrary SQL fragment. Namespace arrays,
ambiguous dotted namespace components and cross-catalog name resolution are not
newly qualified. Projection, filtering and aggregation keep native Iceberg
manifest, field-ID and deletion semantics. No PyIceberg runtime dependency or
Parquet-glob fallback is introduced.

An omitted snapshot ID delegates selection to the native transaction. An
explicit positive signed 64-bit integer selects that exact snapshot through
native `AT (VERSION => ...)`; a missing snapshot fails instead of reading the
latest version. Booleans, strings, zero, negative and overflowing IDs are
rejected. Timestamp-based selection and snapshot/schema-evolution combinations
are not part of this interface's qualified matrix.

`IcebergCatalog(connection, name)` wraps an existing READ_ONLY Iceberg
attachment. It rejects writable, missing and other catalog types. Wrappers
retain the native attachment identity; detaching or replacing that attachment
invalidates them even when an alias is reused. Identifier matching follows the
engine's ASCII case-insensitivity and preserves distinct non-ASCII names.
`detach()` leaves the caller's connection open. Relations depend on their
connection and attachment lifetime, following Vane's native relation contract.

The helper checks the connection's captured runner policy, independently of
later environment changes. Ray connections fail before provider or REST access
with `NotImplementedError`; no runner switch or materialization fallback occurs.
Iceberg-owned distributed transport and scan splits remain M5.

### Additional real-service feature matrix

The M4b fixture uses the same source pins, SDK, engine identities, service image
and test-client versions listed above. It adds independently committed v2
Parquet tables and compares exact rows through the public API:

| Feature | Qualified combination |
| --- | --- |
| Snapshot ID | First/latest append snapshots, pre-delete snapshots, explicit rejection of a missing ID |
| Identity partitions | Optional string partition, Unicode and NULL values; full and filtered results |
| Schema evolution | Rename, drop, add nullable columns, reuse an old column name with a new field ID, INTEGER to BIGINT promotion |
| Position deletes | Parquet delete file with spec field IDs and 64-bit row positions; deleted row excluded |
| Equality deletes | One nullable string equality field; NULL matching; later same-value appends survive the older delete sequence |
| Delete-aware execution | Full rows, projected IDs, NULL filter, count/sum and pre-delete snapshot reads |
| Attachment ownership | READ_ONLY enforcement, wrapping, quoted/Unicode aliases, detach/replacement invalidation |

PyIceberg independently reads the partitioned, evolved and position-delete
tables. Its pinned reader does not support equality-delete scans; that fixture
is validated by reading the SDK-produced manifest entries and Parquet field
IDs, then comparing Vane's rows with the expected values and sequence ordering
from the independently authored inputs. Test delete manifests are emitted with
the pinned SDK's v2 Avro writer and committed through its REST client, retaining
the previous manifests and asserting the parent snapshot. Production Vane
never authors metadata JSON or commit requests in this interface.

Delete formats and sequence applicability follow the
[Apache Iceberg specification](https://iceberg.apache.org/spec/#delete-formats).
Partition pruning efficiency, transform/spec evolution, nested evolution,
multi-column/partitioned equality deletes, deletion vectors, other format
versions, cloud storage, other platforms and other REST services are unqualified.

Run the API contracts before the base gate, using the service settings above:

```bash
scripts/run_installed_pytest.sh tests/fast/test_iceberg_catalog.py
scripts/run_installed_pytest.sh tests/fast/test_gravitino_iceberg_live.py tests/fast/test_iceberg_catalog_live.py -m external_service
scripts/run_release_tests.sh
```

## Compatibility changes and next milestones

The pristine Iceberg build fails in its MERGE planner because Vane's logical
INSERT/UPDATE/DELETE constructors require `ClientContext`. The compatibility
patch passes the context already owned by that planner. Successful compilation
does not qualify any write operation; acceptance attaches READ_ONLY.

Upstream `IcebergScanSerialize` and `IcebergScanDeserialize` explicitly throw
`NotImplementedException`. Vane's native local path executes the live plan
directly. Ray requires an Iceberg-owned transport and split protocol; inherited
Parquet callbacks cannot establish that contract for Iceberg bind state.

M4b qualifies the local read interface and the explicit feature matrix above.
M5 must
freeze a snapshot per execution, serialize the relevant Iceberg state, assign
scan splits, and preserve field IDs and deletion semantics on workers. Each
unsupported feature must fail explicitly. Vane can implement these adapters
against the fixed source without waiting for upstream changes.

This fixture does not qualify v1/v3 tables, production object stores, distributed
reads, catalog writes, authentication extensions, credential vending or refresh.
