# Gravitino Iceberg provider qualification

M4a tracks an independently built Iceberg provider and native local reads
through Gravitino's Iceberg REST service. The tracking issue is
[#1064](https://github.com/AstroVela/vane/issues/1064); production wheel
publication remains [#619](https://github.com/AstroVela/vane/issues/619).
M4a established the provider qualification. M4b adds the local Python interface
below, tracked in [#1066](https://github.com/AstroVela/vane/issues/1066).
M5a adds the narrow frozen-snapshot transport described below, tracked in
[#1068](https://github.com/AstroVela/vane/issues/1068). M5b adds file assignments
and the narrow Ray path, tracked in
[#1073](https://github.com/AstroVela/vane/issues/1073). M5c1 adds identity
partition transport, tracked in
[#1075](https://github.com/AstroVela/vane/issues/1075). M5c2a adds known
transformed-partition reads, tracked in
[#1077](https://github.com/AstroVela/vane/issues/1077). M5c2b adds supported
flat schema evolution, tracked in
[#1079](https://github.com/AstroVela/vane/issues/1079).
M5c3 adds committed v2 position/equality delete reads, tracked in
[#1081](https://github.com/AstroVela/vane/issues/1081).
M3's general Iceberg capability flags remain conservative; this record covers
the explicitly qualified provider, service, runner and feature combinations.

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
acceptance cases passed at M4a. Its historical result qualifies the following narrow matrix:

| Operation or feature | Result |
| --- | --- |
| Source build and signed local provider resolution | Passed; exact Avro dependency loads before Iceberg |
| Existing REST table, format v2, unpartitioned Parquet | Passed; all 4101 independently verified rows match |
| Uncommitted Parquet file beside committed data | Excluded from the Iceberg result |
| Projection, tested numeric filters and aggregation | Passed |
| Empty table | Preserves BIGINT/VARCHAR schema with zero rows |
| READ_ONLY attachment | INSERT rejected and independently observed snapshot unchanged |
| Scan-plan serialization at M4a | Explicitly rejected; M5a's separate scope is described below |
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
Then apply `iceberg-vane-frozen-scan.patch`, `iceberg-vane-file-scan.patch`,
`iceberg-vane-partitions.patch`, `iceberg-vane-transforms.patch` and
`iceberg-vane-schema.patch`, followed by `iceberg-vane-deletes.patch` and
`iceberg-vane-s3.patch` from the
same directory, in that order. The SDK
instructions give the ordered commands and each patch is independently retained.

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
rejected. Timestamp-based selection remains unqualified. Snapshot/schema
evolution combinations are covered by M5c2b and M5c3 below.

`IcebergCatalog(connection, name)` wraps an existing READ_ONLY Iceberg
attachment. It rejects writable, missing and other catalog types. Wrappers
retain the native attachment identity; detaching or replacing that attachment
invalidates them even when an alias is reused. Identifier matching follows the
engine's ASCII case-insensitivity and preserves distinct non-ASCII names.
`detach()` leaves the caller's connection open. Relations depend on their
connection and attachment lifetime, following Vane's native relation contract.

Relations retain the connection's captured runner policy, independently of
later environment changes. Attachment and metadata inspection run on the
coordinator for either runner. M4b and M5a restricted this helper to local
connections; M5b admits Ray connections for the narrower table semantics
[qualified below](#file-assignments-and-ray-m5b).

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

## Frozen snapshot transport (M5a)

This stage added bind serialization and an independently owned worker bind to
the source-pinned provider. At M5a, `IcebergCatalog` kept its public local-runner
guard. Native worker execution was exercised directly with Ray initialization
forbidden; real Ray scheduling is qualified separately in M5b below.

The transport accepts format-v2, unpartitioned, committed Parquet tables whose
metadata contains exactly one flat, field-ID-mapped schema. Data-file paths
must be absolute local paths or `file:///` URIs. Partitioned tables, schema
evolution/name mappings, nested fields, delete manifests, uncommitted writes,
alternate Parquet schemas, encryption and row-lineage virtual columns are
outside this stage. Their qualified M4b local reads continue to use the live
Iceberg reader. Transport rejects unsupported state explicitly rather than
silently substituting a directory glob or rereading metadata.

Binding selects a snapshot once. Serialization freezes its table UUID,
snapshot-presence flag and ID, schema ID, output schema, field IDs/defaults,
reader options and exact manifest-selected file paths. Restoring that state
does not replay the catalog attachment, request newer metadata, open manifests
or retain a client context, table entry or mutable coordinator cache. It uses
the existing Parquet reader for file decoding while preserving Iceberg field
mapping. The independently loaded provider and its Avro dependency still pass
the existing exact engine/provider identity checks.

The native bind format has version `1`. Physical translation constructs a
separate worker bind containing only owned schema/options and snapshot
identity, with an empty file list and an unassigned role. It selects those
members explicitly; `FunctionData::Copy()` cannot serve as a worker-state
fallback. The detached reader is marked separately from the live Iceberg
reader so live-catalog optimizer hooks cannot reinterpret its file list.

M5a uses the existing distributed scan callbacks to assign the complete
snapshot as one work item. This is the minimum executable transport path,
before file-level partitioning. Its codec is
`vane.iceberg.frozen-whole-scan`, version `1`, with canonical split ID
`snapshot`. An assignment carries its own version, frozen identity and file
paths. Decoding checks identity, version, split count/ID, path policy and
trailing bytes before replacing the worker's file list. A missing assignment
fails; an explicit empty assignment clears files and produces no rows. An
empty table retains its schema and uses the framework's explicit empty split.

M5a retains M4b commit `0606e2a038b4007a04bf34541e76af0462cff113` and
integrates upstream main `8ef29bee9d5d1ada3d7d5efe77302a296ff6576a`.
The table above records the historical M4a/M4b engine. M5a's current engine
tree SourceID is `aaa05c82f52d494cf53b3199fe8b2711395aae4d` and its fork
revision is `8ef29bee9d5d1ada3d7d5efe77302a296ff6576a`; the Iceberg/Avro
source pins, SDK profile and service identity are unchanged. Both providers
must be rebuilt for that engine identity. The additional frozen-scan patch has SHA-256
`b6f1ff2b097c176744b0bd1c9dd0721ac567247b0fece83e82f5666f28988be9`.
The existing local compatibility patch is unchanged. M5a applied these two in
order; M5b adds a third patch after them.

The provider explicitly serializes field IDs and constant defaults: the
generic reader-bind serializer does not include its schema or mapping mode.
Before freezing paths it fully enumerates the selected manifests, since the
pinned Iceberg list's `GetAllFiles()` alone returns only materialized entries.
The loadable artifact links Parquet internally to preserve Vane's private
native-module symbol boundary; the engine itself requires no change.

### Acceptance and reproduction

The M5a tests use the fixed real-service and independent SDK inputs above.
They close the original Vane connection and the forwarding REST endpoint,
then temporarily move all fixture metadata directories out of their original
locations while leaving data files accessible. A restored worker must still
return the independently verified rows, projection, filters and aggregation,
and preserve the empty schema. A separate independently committed append
must not appear in the frozen result. Nonempty foreign-table assignments are rejected,
explicit empty assignments clear nonempty workers, and missing assignments
fail before execution. Existing partition/evolution/delete fixtures must
continue to read locally while their transport fails explicitly.
Independent v1 and nested-schema metadata fixtures also exercise the admission
guards. The native task API reports positional `col_N` result names for all
sources; this stage checks empty-result arity and BIGINT/VARCHAR types there.
The existing M4b suite separately verifies public Relation column names.

```bash
scripts/run_installed_pytest.sh tests/fast/test_gravitino_iceberg_live.py tests/fast/test_iceberg_catalog_live.py tests/fast/test_iceberg_frozen_scan_live.py -m external_service
scripts/run_installed_pytest.sh tests/fast/test_iceberg_catalog.py
scripts/run_release_tests.sh
```

The same two opt-in service variables used above are required. This test
module's forwarding endpoint contacts the real provisioned Gravitino service;
it does not synthesize catalog responses. Stopping that endpoint tests the
address captured by the plan. A separate qualification run must stop the
actual owned service and restore in a fresh process before claiming that
stronger acceptance condition.

On 2026-10-23, the 46 real-service cases (15 M5a, 6 provider and 25 local API)
passed on the fixed Linux/Python/SDK profile. The 20 local catalog contracts
and two affected generic native transport contracts also passed. A separate
run stopped the actual test-owned Gravitino container after all source Vane
connections closed and an independent append committed. A fresh Python process
then hid the fixture metadata directories and passed six restored queries:
all rows, projection, filter, aggregation, empty-table types and an explicit
old snapshot. The test-owned fixture files and stopped service were removed
after the run. The
standard base release gate remains required and is reported with the change.
These were the initial M4b-engine results. Requalification for the synchronized
engine above, including the complete base gate, is recorded in
[#1069](https://github.com/AstroVela/vane/pull/1069).

## File assignments and Ray (M5b)

M5b keeps M5a's frozen bind format and table guards. It uses the existing scan
callback and FTE scheduler APIs; no engine or runner-specific scan path is
introduced. Fault injection exposed and fixes one shared FTE dependency bug:
when a replacement worker already has the fragment cached, the new attempt
must clear the original worker's registration dependency instead of awaiting
its dead actor's ObjectRef. A separate scheduler regression covers both
pending and cached replacement registration. The patch `packages/vane-iceberg-provider/patches/iceberg-vane-file-scan.patch`
applies after the unchanged M4a and M5a patches. Its SHA-256 is
`784c3f5f082f2b454b31be3848d9b219a751ae03aed68180f01b3251e27b9988`. The engine, Iceberg, Avro and SDK identities remain those recorded
for M5a above. Rebuild and repackage the signed Iceberg provider; a previously
built whole-scan provider does not implement this protocol.

Each committed data-file entry produces one split, with canonical ID `file-N`
where `N` is its ordinal in the frozen file list. IDs and assignments are
independent of worker-slot and target-split-count hints. Planning fully expands
the selected manifests, excludes neighboring uncommitted files and rejects
duplicate data paths. Row groups and file-size balancing remain future work;
unknown byte/cardinality estimates do not justify truncating or prebatching
the file list.

The new codec is `vane.iceberg.frozen-file-scan`, version `1`. Its versioned
payload contains the table UUID, snapshot-presence flag/ID, schema ID, file
ordinal and one absolute local path. Worker application validates the codec
through the shared scan envelope and validates every payload's version,
identity, ID, path and trailing bytes. It rejects duplicate ordinals or paths
within an assignment. Decoding first builds temporary owned state, then
atomically installs only the assigned subset in ordinal order. Missing
assignments fail and explicit empty assignments clear all files. The old
whole-scan codec is a different capability and cannot be mixed with this one.
Retries use the original frozen identity and file assignments.

`IcebergCatalog.attach()` and `.table()` now admit Ray connections. Loading
and validating the matching signed provider, READ_ONLY enforcement, identifier
quoting and attachment-lifetime checks are the same as the local interface.
Catalog operations stay on the coordinator; workers restore provider state
and frozen scans without replaying REST attachments. The connection's runner
policy is preserved, with no implicit runner switch.

The Ray matrix is deliberately narrower than the local matrix:

| Feature | Ray scope |
| --- | --- |
| Table and storage | Committed, unpartitioned format-v2 Parquet; one flat field-ID schema; absolute local paths shared by every worker |
| Reads | Rows, projection, tested filters, BIGINT aggregation, empty-table schema, explicit snapshot ID |
| Scheduling | One independently schedulable split per committed file; more files than configured worker slots |
| Frozen replay | Original connection and captured REST endpoint closed; newer independent append and unavailable metadata/manifests do not change the selected rows |
| Recovery | Real production worker actors, disjoint assignments, worker loss and replacement attempt, cancellation and query/session cleanup |
| Provider replay | Exact signed Iceberg/Avro dependency closure; mismatched Iceberg digest rejected before fragment publication |
| Extended table semantics | Partitions, evolution, nested fields and position/equality deletes remain explicit transport errors; qualified local reads keep working |

Ray's existing result-type policy still applies. For example, `sum(BIGINT)`
returns HUGEINT by default; use a checked BIGINT cast for a result known to fit,
or configure Vane's existing lossless Arrow conversion. This milestone does not
change result-type conversion.

The independent fixture uses nine committed files totaling 4101 rows, NULL and
Unicode values, an empty table and an adjacent uncommitted Parquet file. The
native assignment tests compare individual and merged splits against
PyIceberg. The real-Ray suite exercises the public relation API and two
production `RayWorkerActor` processes on one host through the existing FTE
protocol. The controlled worker-loss case leaves the source stream open,
kills its first actor, finishes the replacement attempt and checks exact rows
without duplicate output. Cancellation checks terminal task state, released
fragment ownership and closed session snapshot databases. This is shared-path,
same-host qualification, not qualification of a remote storage deployment.

Reproduce after the ordered provider build and signed-wheel installation:

```bash
export VANE_TEST_GRAVITINO_ICEBERG_URL=http://127.0.0.1:9001/iceberg
export VANE_TEST_GRAVITINO_ICEBERG_SHARED_DIR=/absolute/shared/test-warehouse
scripts/run_installed_pytest.sh tests/fast/test_iceberg_frozen_scan_live.py \
  tests/fast/test_iceberg_catalog_live.py -m external_service
scripts/run_installed_pytest.sh tests/fast/test_iceberg_ray_scan_live.py \
  -m external_service
scripts/run_installed_pytest.sh tests/fast/test_iceberg_catalog.py
scripts/run_release_tests.sh
```

The Ray suite uses the repository cluster fixture and normal object-store
sizing. It creates and kills only its owned actors. Service namespaces and
warehouses remain unique per fixture and are removed after the tests.

## Identity partition transport (M5c1)

This first M5c delivery adds identity-partitioned format-v2 local Parquet to
both native frozen execution and the public Ray relation interface. It keeps
M5b's coordinator/worker boundary and per-file scheduling. The source/SDK/engine
pins above are unchanged. Apply the fourth patch
`packages/vane-iceberg-provider/patches/iceberg-vane-partitions.patch` after the
three immutable earlier patches. Its SHA-256 is `58ab2d9fa4980f7917839c3dc9b9da5971eb3acf1468eb373205f89636347c48`.
The inherited reader shares its existing value decoder with the frozen
adapter, preserving the live local path and exact attribution.

The important case is a migrated file that omits an identity partition column.
Iceberg's [column projection rules](https://iceberg.apache.org/spec/#column-projection)
select physical fields by ID and recover a missing identity field from its
manifest partition record. Directory layout and physical field names do not
supply that value. A present physical field wins; NULL in a manifest remains a
typed NULL. The remaining schema-default behavior continues to use the native
field-ID mapper.

The frozen bind format, assignment payload and `vane.iceberg.frozen-file-scan`
codec now use version **2**. A frozen file owns its absolute path, selected
partition-spec ID, source field IDs and typed identity values. The coordinator
fully enumerates the selected files and resolves each file's own manifest/spec.
Historical unpartitioned and identity specs may coexist in one snapshot,
including spec renaming and explicit older-snapshot selection. This does not
qualify general schema evolution or transformed partition specs.

No partition constant enters the split-free worker bind. Worker application
decodes all assigned file state into temporary owned storage, checks versions,
snapshot identity, paths, file IDs, nonnegative spec IDs, unique known source
field IDs, matching vector lengths and exact value types, then atomically
replaces its assignment. Empty assignments clear both files and constants.
Version-1 frozen binds/assignments are explicitly incompatible and rejected;
rebuild and repackage the matching signed provider rather than mixing stages.

Reader initialization uses the assigned file's own constants only for source
field IDs absent from that Parquet schema. The existing native mapper applies
constant filters and field projection. This also handles partition-only
projections, count-only scans and no-match/empty results. Restoring a plan or
retrying a task cannot reopen REST, JSON metadata or manifests to recover
partition state.

| Feature | M5c1 qualification |
| --- | --- |
| Identity values | NULL and Unicode strings; date, decimal and microsecond timestamp values, including pre-epoch and negative values |
| Physical columns | Present under different physical names but matching field IDs; omitted identity fields recovered from manifest records |
| Path layout | Opaque files inside a deliberately misleading partition directory; adjacent uncommitted Parquet excluded |
| Partition specs | Mixed historical unpartitioned/identity specs and renamed partition fields, each resolved per file |
| Reads | All rows, partition-only projections, NULL/constant/mixed predicates, counts, typed aggregates, empty tables and explicit old snapshots |
| Frozen execution | Source connection and captured REST endpoint closed; metadata directories unavailable; a later independent partition append leaves the frozen result unchanged |
| Real Ray | Public relations, two production worker processes with disjoint files, worker loss/replacement, selected-attempt output and cancellation/query/session cleanup |
| Admission | Malformed IDs/types/versions rejected; exact signed Iceberg/Avro replay and the existing unpartitioned/native protocols retained |

The fixture writer uses PyIceberg's snapshot transaction API and PyArrow
Parquet files with explicit field IDs. The same independent SDK scans verify
the committed rows before Vane binds anything. Nine-file tables exceed the
configured one/two worker slots; the mixed-spec table retains seven files.
Native and real-Ray suites share the fixture and FTE helpers while retaining
separate pytest processes. The local M4b partition/schema/delete matrix and
M5a/M5b unpartitioned transport and Ray regressions remain required.

Reproduce with the service and signed provider setup described above:

```bash
scripts/run_installed_pytest.sh tests/fast/test_iceberg_partition_scan_live.py \
  tests/fast/test_iceberg_frozen_scan_live.py tests/fast/test_iceberg_catalog_live.py \
  tests/fast/test_gravitino_iceberg_live.py -m external_service
scripts/run_installed_pytest.sh tests/fast/test_iceberg_partition_ray_live.py \
  tests/fast/test_iceberg_ray_scan_live.py -m external_service
scripts/run_release_tests.sh
```

Identity partition support completes only this first M5c delivery. M5c2a below
extends the transformed-partition matrix. Nested fields, schema evolution/name
mappings, delete manifests, writes, remote storage and authentication retain
the existing boundaries.

## Transformed partition reads (M5c2a)

The fifth ordered patch, `iceberg-vane-transforms.patch`, admits the partition
transforms already understood by the pinned provider: `year`, `month`, `day`,
`hour`, `bucket[N]`, `truncate[W]` and `void`, alongside identity. Its SHA-256
is `674bbcb52d7c0bfdaf335e0e438ad8d036d69d1253a3f8d70428d6ac1286d1a7`.
The four earlier patches, source/SDK pins, engine identities and dependency
closure are unchanged. A small inherited-reader fix keeps a later bucket or
time field from overwriting the identity lookup for the same source column.
It retains the existing MIT AND Apache-2.0 attribution. No additional Local/Ray
runner implementation is introduced.

The coordinator still resolves each selected file through its own manifest's
partition spec. It freezes only identity fields as source-column constants.
A day, bucket, truncated prefix or void result cannot reconstruct the source
value and never enters that constant map. Physical fields continue to resolve
by field ID; omitted identity fields retain M5c1's typed recovery behavior,
including a source that also has a bucket transform in the same spec.

The frozen bind, assignment and file-split codec stay at version **2** because
their owned data structure is unchanged. Worker binds stay split-free, received
assignments retain atomic validation, and retries replay the original file
selection without REST or manifest access. Exact signed provider identity is
still required. No persisted-state compatibility with a different provider
digest is implied by the unchanged codec version.

This delivery qualifies correct reads and original-value row predicates. It
does not introduce transformed partition-pruning state in worker binds or claim
new pruning performance. Restored scans retain normal Parquet field mapping and
row filtering. Unknown transform names follow the pinned provider's explicit
parser rejection. M5c2b below extends flat schema evolution; explicit name
mappings and delete manifests remain rejected by the frozen adapter.

| Feature | M5c2a qualification |
| --- | --- |
| Time | Year/month/day/hour of timestamps, day of dates, NULL, microseconds, leap-day/month/year/hour boundaries and times before 1970 |
| Bucket | Multiple bucket counts over integer, Unicode string and decimal source columns |
| Truncate | Negative integer and decimal boundaries; Unicode prefixes and NULL values |
| Void | Non-null original date values remain readable despite NULL partition records |
| Mixed specs | Omitted identity source also bucket-partitioned; time-transform and bucket-parameter changes in three historical specs, including old snapshots |
| Files | Opaque paths under a misleading partition directory, committed files exceeding worker slots, adjacent uncommitted Parquet and empty tables |
| Native/local | Detached frozen execution, projections, original-value filters, counts, empty assignments, public catalog relations and independent append |
| Real Ray | Public relations, frozen replay after append/closed REST, two production workers with disjoint files, replacement without duplicate rows and cancellation/session cleanup |

The SDK fixture writer now applies the SDK transform to its native partition
representation before committing each record. It continues to omit only identity
source columns and uses physical field IDs when renaming stored columns. SDK
scans independently verify all current and initial-snapshot rows before Vane
binds them. Filtered native references apply independent PyArrow predicates to
the committed SDK scan, avoiding the pinned SDK's nullable partition-filter
evaluation error while retaining the complete NULL cases. The nine
nonempty tables have 3 to 27 committed files for one/two worker slots; the
mixed historical-spec table has 27 files across three commits. The same-host,
absolute shared-local-path boundary from M5b remains in force.

Run the affected installed suites in separate native and real-Ray processes:

```bash
scripts/run_installed_pytest.sh tests/fast/test_iceberg_transform_scan_live.py \
  tests/fast/test_iceberg_partition_scan_live.py tests/fast/test_iceberg_frozen_scan_live.py \
  tests/fast/test_iceberg_catalog_live.py tests/fast/test_gravitino_iceberg_live.py -m external_service
scripts/run_installed_pytest.sh tests/fast/test_iceberg_transform_ray_live.py \
  tests/fast/test_iceberg_partition_ray_live.py tests/fast/test_iceberg_ray_scan_live.py -m external_service
scripts/run_installed_pytest.sh tests/fast/test_iceberg_catalog.py
```

The user explicitly requested affected tests only for this delivery; a complete
base release-gate result is not part of this qualification. Formatting, source
headers, the five ordered patches, license inventory, archive contents and the
untouched original workspace are checked separately.

## Flat schema evolution (M5c2b)

The sixth ordered patch, `iceberg-vane-schema.patch`, admits retained flat schema
history while preserving the selected schema's names, order, field IDs, types
and typed NULL defaults. Its SHA-256 is
`f19ce28ed74c747af977969c9e3e5c20b9708ddb72604501a338b479e6e3336e`.
The five predecessor patches, source/SDK pins and engine identities are
unchanged. The frozen bind/assignment/file codec remains version **2**: schema
IDs and field/default vectors were already owned serialized state. Restored
default types are now checked against their frozen target fields.

Schema selection follows the bound catalog entry. Explicit old snapshots retain
their historical schema. A metadata-only schema update uses the new schema even
when its data snapshot is unchanged; assignment identity includes both snapshot
and selected schema IDs. Workers reject a different schema's assignment for the
same table/snapshot rather than interpreting its fields in another order.

Reads continue to follow Iceberg's [field-ID projection rules](https://iceberg.apache.org/spec/#column-projection).
Rename/reorder preserve IDs. Dropping and re-adding a name creates a distinct
field: old physical columns and partition values must not populate the new
field. Missing optional fields use typed NULL for this format-v2 profile.
The pinned Gravitino service rejects non-null initial defaults in v2; these
require v3 and remain outside this delivery. Native field/default mapping is
reused, without adding runner-specific schema interpretation.

The inherited manifest decoder now preserves decoding types for legacy
partition sources removed from the selected schema. It consults retained
schema history and widens compatible historical types before decoding those
values; a nonempty Avro partition struct must not be projected to SQL NULL.
File and manifest filter lookup also skips absent legacy sources conservatively
instead of indexing a missing selected field. This prevents old partition
statistics from filtering a newly added field that reuses the old name.
The frozen adapter omits those historical-only constants from assignments,
requiring their IDs to exist in retained schema history; selected fields still
receive only their own identity constants.

| Feature | M5c2b qualification |
| --- | --- |
| Field evolution | Add/drop/rename/reorder, two reused column names with fresh IDs, Unicode, NULL and boolean fields |
| Promotions | `int` to `long`, `float` to `double`, decimal precision 8 to 12 to 18 with scale 2 unchanged; mixed old/new physical files |
| Identity sources | Omitted integer/float/decimal fields across promotion, including an integer source also bucket-partitioned |
| Transformed sources | Bucket/truncate fields across rename and valid promotion, with original source-value reads |
| Dropped partition source | Historical identity spec, source deletion, new field reusing the name, and explicit old-schema reads |
| Schema-only updates | Changed names/order/new nullable field with the same data snapshot; old/current schema assignment mismatch rejected |
| NULL defaults | Old files missing added fields, explicit NULL and non-null values in new files, malformed typed-NULL defaults rejected |
| Empty/lifecycle | Evolved empty table, explicit empty assignment, disjoint mixed-schema subsets, independent schema change/append after freeze, real-Ray worker replacement and cancellation |

PyIceberg authors the schema changes and Parquet/manifest commits. Independent
reference scans cover current and initial/intermediate snapshot schemas before
Vane binds them. The pinned SDK's omitted identity FLOAT-to-DOUBLE scan loses
precision for new DOUBLE values. That fixture therefore uses a separate
physical-column control table with identical input rows, field IDs and schema
history. Reference selection verifies snapshot schema IDs, row counts and row
IDs before using that table. Values beyond float32 precision remain in the
fixture and in dedicated native/Ray predicate checks. No expected value is
derived from Vane's output. Arrow predicates use explicitly typed decimal
thresholds and SQL NULL semantics.

Seven nonempty tables have 3 to 27 committed files, alongside an evolved empty
table. Physical paths are opaque under a misleading directory, with uncommitted
orphans excluded. Metadata and the captured REST endpoint are unavailable during
detached native/Ray replay. A later independent schema update plus append must
leave the frozen rows, field order and target types unchanged.

Run only the affected installed suites, keeping native and real-Ray execution
in separate pytest processes:

```bash
scripts/run_installed_pytest.sh tests/fast/test_iceberg_schema_scan_live.py \
  tests/fast/test_iceberg_transform_scan_live.py tests/fast/test_iceberg_partition_scan_live.py \
  tests/fast/test_iceberg_frozen_scan_live.py tests/fast/test_iceberg_catalog_live.py \
  tests/fast/test_gravitino_iceberg_live.py -m external_service
scripts/run_installed_pytest.sh tests/fast/test_iceberg_schema_ray_live.py \
  tests/fast/test_iceberg_transform_ray_live.py tests/fast/test_iceberg_partition_ray_live.py \
  tests/fast/test_iceberg_ray_scan_live.py -m external_service
scripts/run_installed_pytest.sh tests/fast/test_iceberg_catalog.py
```

The full base release gate is excluded by the user's validation scope. Keep the
six-patch, SPDX/license, archive and original-workspace checks. Qualification
remains same-host shared absolute local paths, committed format-v2 flat Parquet
schemas and history, and no delete manifests. Nested history and explicit name
mappings retain explicit guards. Remote storage, v3/non-null defaults,
authentication and writes remain separate. Ray's existing Arrow result-type
policy remains in force.

## Position and equality deletes (M5c3)

The seventh ordered patch, `iceberg-vane-deletes.patch`, freezes committed v2
Parquet delete-file descriptors with each independently assignable data file.
Its SHA-256 is
`4863d1f57d116882698539bb0a911e968584b2cadd5345b2624ed7a6df4cf56a`.
All six predecessor patches, provider/SDK pins and engine identities are
unchanged. This stage follows Iceberg's [scan planning](https://iceberg.apache.org/spec/#scan-planning)
and [delete formats](https://iceberg.apache.org/spec/#delete-formats).

The coordinator captures the selected snapshot's data sequence, record count
and complete partition key for each data file, plus only the delete files that
apply to it. A descriptor owns its exact path, content, data sequence, record
count/size, spec/partition tuple, optional referenced data file and equality
field IDs. File paths are compared exactly and case-sensitively. Workers
require both the selected data and delete files to remain accessible at the
same absolute local paths, following Iceberg's immutable-file contract. They
never reread catalog metadata/manifests or discover newer delete files. A
missing file fails explicitly rather than returning undeleted rows.

| Delete kind | Applicability and execution |
| --- | --- |
| Position | Matching partition spec/values, data sequence less than or equal to delete sequence, optional referenced file match, then exact target path and zero-based row position |
| Equality | Data sequence strictly less than delete sequence, matching spec/partition or an unpartitioned global delete, and NULL-safe equality of every selected field ID in a delete row |
| Multiple files/rows | Union of applicable deletes, duplicate positions/keys allowed; removed manifest entries and uncommitted files do not enter the frozen state |
| Evolution | Historical specs retain their own keys; dropped equality fields remain hidden scan inputs; reused names have distinct IDs; added fields missing from old files read as NULL |
| Type promotion | Old equality values use the selected field type, including INTEGER to BIGINT and same-scale decimal precision widening; a later dropped field keeps its widest retained type |

Frozen bind, assignment and file codec versions are now **3**. Earlier versions
are rejected; the generic distributed callback protocol stays at **1**. Worker
binds retain the equality field-ID/type declarations but no coordinator files
or decoded delete cache. Assignment decoding validates all file state before
installing it, including nonnegative sequences/counts, partition consistency,
applicability, known unique equality IDs and conflicting delete identities.
Explicit empty assignments clear all files. Physical delete fields are resolved
by Parquet field IDs and checked for compatible types; positions must be
nonnegative and below the referenced data file's record count. Delete record
counts must agree with the manifest.

Each worker assignment decodes a selected delete file once into owned immutable
cached data. Position filters use original Parquet row positions across vector
and row-group boundaries. Equality rows use hash buckets followed by exact
NULL-safe tuple comparisons. A provider optimizer guard preserves required
fields through projection and count pruning without exposing dropped fields in
the public relation. Replacement attempts rebuild their own state from the same
descriptors. No generic Local/Ray scheduler or protocol implementation changes
are needed.

The live local reader receives the corresponding fixes: NULL-safe equality and
partition comparison, whole-file partition rejection, hidden-field indexes
that are column positions rather than field IDs, promoted delete constants,
position field-ID decoding, exact target paths and sequence/partition-scoped
position filters. Existing write/delete-authoring behavior is not qualified by
these read-path changes.

Ten independent fixture tables cover large multi-row-group positions, duplicate
positions and case-distinct paths, older/equal/newer sequence boundaries,
nullable and multi-column equality keys, NULL partition isolation, global
deletes and spec evolution, dropped/reused/added equality fields, INTEGER/decimal
promotion with date/boolean keys, all-deleted output and removal of active delete
manifests. Current and pre-delete/intermediate snapshots are compared separately.
PyIceberg writes Parquet/Avro and commits through the real REST service; fixture
checks verify the emitted field IDs, content, partitions and data/file sequences.
The pinned SDK cannot execute equality-delete scans. A separate reference
evaluator therefore reads its manifests and Parquet through Arrow and applies
the specification's predicates. An additional SDK scan checks the supported
position-delete fixture. No expected result comes from Vane.

Related installed tests cover public local/Ray rows, projection, filters,
aggregation and old snapshots; closed-REST/unavailable-metadata replay; later
independent delete/append commits; disjoint assignments; worker replacement;
cancellation and query/session cleanup. Negative cases include missing delete
files, invalid positions, malformed owned delete descriptors, stale wire
versions and an invalid later assignment before any data file is opened.

```bash
scripts/run_installed_pytest.sh tests/fast/test_iceberg_delete_scan_live.py \
  tests/fast/test_iceberg_schema_scan_live.py tests/fast/test_iceberg_transform_scan_live.py \
  tests/fast/test_iceberg_partition_scan_live.py tests/fast/test_iceberg_frozen_scan_live.py \
  tests/fast/test_iceberg_catalog_live.py tests/fast/test_gravitino_iceberg_live.py -m external_service
scripts/run_installed_pytest.sh tests/fast/test_iceberg_delete_ray_live.py \
  tests/fast/test_iceberg_schema_ray_live.py tests/fast/test_iceberg_transform_ray_live.py \
  tests/fast/test_iceberg_partition_ray_live.py tests/fast/test_iceberg_ray_scan_live.py -m external_service
scripts/run_installed_pytest.sh tests/fast/test_iceberg_catalog.py
```

The user's validation scope remains affected tests only, in separate native and
real-Ray pytest processes with normal object-store sizing. Verify the seven
ordered patches against compiled sources, source/SDK notices and SPDX, formatting,
source archive and untouched original workspace. Qualification remains Linux
x86-64, same-host shared absolute local paths and committed v2 flat Parquet
schemas/history. Nested/name mappings, Puffin/deletion vectors/v3, remote
storage, authentication, writes and production provider delivery remain separate.
Ray's existing Arrow result-type policy still applies.

## Anonymous S3/MinIO (M5d1)

The eighth ordered patch, `iceberg-vane-s3.patch`, admits canonical
`s3://bucket/object` locators for frozen data, delete descriptors and position
delete targets. Its SHA-256 is
`cce75954d8b71d1109b09feec5b063a71e749923f846543d0d7e61c58809e73a`.
All seven predecessor patches and the M5c3 engine/provider/SDK pins are
unchanged. Frozen bind/assignment/file codec 3 and generic callback protocol 1
are unchanged: the URI string already belongs to the frozen assignment.

Paths retain their exact bucket/key bytes, including Unicode; no local path
rewrite or Hive-directory inference occurs. The admitted subset requires a
lowercase DNS-style bucket of 3–63 characters and a nonempty key. Control
characters, userinfo, ports, query strings, fragments and other remote schemes
are rejected before assignment I/O. Keys containing `?` or `#`, alternative
S3 schemes and other cloud filesystems remain outside this qualification.

`IcebergCatalog.attach` explicitly uses `ACCESS_DELEGATION_MODE 'none'` in
addition to `AUTHORIZATION_TYPE 'none'` and `READ_ONLY`. This disables the
pinned provider's default credential-vending request. Existing temporary or
persistent DuckDB secrets are not imported into frozen worker state. Storage
settings use the existing connection snapshot and worker database isolation;
M5d1 adds no runner protocol or credential mechanism.

For a bucket already readable anonymously, configure the connection explicitly:

```python
import vane
from vane.catalog import IcebergCatalog

with vane.connect(config={
    "autoinstall_known_extensions": "false",
    "autoload_known_extensions": "false",
    "allow_persistent_secrets": "false",
    "s3_endpoint": "minio.example.test:9000",
    "s3_region": "us-east-1",
    "s3_url_style": "path",
    "s3_use_ssl": "false",  # Match the explicitly provisioned endpoint.
    "s3_access_key_id": "",
    "s3_secret_access_key": "",
    "s3_session_token": "",
}) as con:
    vane.load_installed_extension("iceberg", connection=con)
    lake = IcebergCatalog.attach(con, "lake", endpoint="http://catalog.example.test:9001/iceberg")
    rows = lake.table("analytics", "events").project("id").fetchall()
```

Use `VANE_RUNNER=local` or `ray` before creating the connection. The installed
signed Iceberg/Avro closure must be present for every executing worker. The
endpoint and bucket must be reachable by those workers. This acceptance uses
Linux x86-64 and a single-node Ray cluster; it does not establish multi-node
reachability, deployment or throughput.

The real-service fixture uses the same pinned Gravitino 1.3.0 image as M5c3,
`minio/minio@sha256:14cea493d9a34af32f524e538b8346cf79f3321eff8e708c1e2960462bd8936e`
and `minio/mc@sha256:a7fe349ef4bd8521fb8497f55c6042871b2ae640607cf99d9bede5e9bdf11727`.
Provision an owned bucket with anonymous download access. The independent SDK
and the REST server need fixture-only write credentials to author commits;
those credentials are never supplied to Vane. Configure Gravitino's memory
backend, `ResolvingFileIO`, S3 warehouse, endpoint, region and path-style access
using the [1.3.0 service settings](https://gravitino.apache.org/docs/1.3.0/iceberg-rest-service/).
The server's endpoint may use the container-network hostname while the test
client uses a host-reachable endpoint. The SDK explicitly disables access
delegation too, so no credential provider is required for the REST service.

Set `VANE_TEST_GRAVITINO_ICEBERG_S3_CONFIG` to an untracked, private JSON file
(mode 0600) with these fields:

```json
{
  "endpoint": "http://127.0.0.1:9001/iceberg",
  "s3_endpoint": "http://127.0.0.1:9000",
  "s3_region": "us-east-1",
  "bucket": "owned-test-bucket",
  "admin_key": "fixture-writer-key",
  "admin_secret": "fixture-writer-secret"
}
```

Install the pinned test-only requirements from the provider SDK profile. Do
not export fixture writer keys as AWS environment variables: Ray inherits its
job environment. The fixture creates a unique namespace/object prefix and
removes only that prefix and its catalog entries. Its local directory contains
control/audit files, never a dataset mirror. PyIceberg commits all metadata,
manifests, data and deletes directly to S3. Independent SDK scans check the
supported matrix; the M5c3 manifest/field-ID Arrow evaluator checks equality
deletes. The seven-table matrix includes unpartitioned and empty tables,
identity partitions with omitted physical columns, flat schema evolution,
position deletes, nullable multicolumn equality deletes and an all-deleted table.

A separate HTTP forwarding process sends every Vane object request to real
MinIO. It rejects signed requests and object listing, records range reads, and
can deny metadata or selected data/delete objects. Tests cover public local and
Ray rows/projection/filters/counts/history, detached replay after REST closure
and later commits, disjoint file assignments, missing-object failures, worker
replacement, cancellation, query/session cleanup and per-query endpoint
isolation within one session. Frozen payloads exclude fixture writer keys.

```bash
scripts/run_installed_pytest.sh tests/fast/test_iceberg_s3_scan_live.py -m external_service
scripts/run_installed_pytest.sh tests/fast/test_iceberg_s3_ray_live.py -m external_service
```

Run these in separate installed-package processes with normal Ray object-store
sizing. Run affected local delete/path and connection snapshot regressions;
the user's validation scope remains related tests only. Verify all eight
ordered patches against the source pin and compiled bytes, license/SPDX,
formatting, archive contents and the untouched original checkout.

## Isolated Ray nodes (M5d2)

M5d2 exercises the M5d1 anonymous S3 reader with a driver/head container and
two worker containers on one Docker host. The head advertises zero CPUs so
native scan actors run on the two worker nodes. Every container has its own
filesystem and mount namespace; no checkout, virtual environment, dataset,
Docker socket or host directory is mounted into it. Gravitino, MinIO and the
object/REST forwarding services also have separate containers. MinIO's private
storage is an owned tmpfs. Only object storage carries the Iceberg dataset.

This gate also fixes a node-loss recovery gap in the Python runner. Ray can
report `ActorUnavailableError` before it confirms that a lost node's actor has
died. The failed-worker prepare barrier now retries that transport response
within its existing 30-second total deadline. Only a successful fence/drain
acknowledgement or confirmed actor death permits retries; application errors,
timeouts, persistent unavailability and failures originating inside a live
actor still fail closed. The scheduler's quiescence criterion is unchanged.

The engine identity, eight ordered Iceberg patches, frozen codec 3 and callback
protocol 1 stay unchanged. The installed runtime is the noneditable
`vane-ai==0.3.0.dev79` base and its exact signed test-only Iceberg/Avro closure.
Qualification helpers are copied into each container separately; they never
replace installed Vane. This Python-only runner fix needs no native rebuild.

The acceptance matrix checks:

- Three distinct Ray NodeIDs/IPs and two distinct worker mount namespaces;
  a driver-only filesystem canary is absent on both workers.
- Exact provider descriptors and artifact hashes on every worker, and no
  fixture writer credentials in worker environments or frozen payloads.
- Public rows, projection, filters, aggregates and old snapshots for the seven
  M5d1 table shapes, with data/delete range requests observed from worker IPs.
- Frozen execution after its REST endpoint is stopped and metadata object
  access is denied; an independent later delete/append commit does not change
  the captured result.
- Hard node affinity and disjoint assignments on both workers, retaining
  position/equality deletes and identity partition constants.
- Actor failure and whole-worker-container failure, retry on the surviving
  node, selected-attempt ownership, query/session cleanup and node rejoin.
- Missing data/delete objects fail; removing Iceberg from one worker fails
  provider admission there. Offline reinstall of the exact wheel and a fresh
  Ray node restore reads without extension downloads.

The object forwarder now records client IPs and accepts an explicit bind
address/port for container networking; its default remains loopback with an
ephemeral port. The REST forwarder has the same optional binding controls.
The shared real-Ray test helper accepts explicit node affinity while retaining
the existing single-node default.

### Reproduce the isolated-node gate

Use a source checkout and the installed launcher from [DEVELOPMENT.md](DEVELOPMENT.md).
Prepare the M5d1 base, Iceberg and Avro wheels through the existing build and
test-provider workflow. The host needs the independent SDK requirements in
`packages/vane-iceberg-provider/requirements-test.txt` and permission to use
Docker. The three service images pinned in M5d1 must already be present;
tests use `--pull=never` and fail explicitly if inputs are missing.

Create a temporary build context with `Dockerfile` and
`runtime-requirements.txt` from `tests/iceberg_multinode/`, a `wheels/` directory
containing exactly the matched base/Iceberg/Avro wheels, and a
`runtime-wheels/` directory containing the fixed runtime dependencies:

```bash
python -m pip download --only-binary=:all: \
  -r tests/iceberg_multinode/runtime-requirements.txt \
  --dest /absolute/path/to/build-context/runtime-wheels
docker build --pull=false --tag vane-iceberg-multinode:qualification \
  /absolute/path/to/build-context
export VANE_TEST_ICEBERG_MULTINODE_IMAGE="$(docker image inspect \
  --format '{{.Id}}' vane-iceberg-multinode:qualification)"
scripts/run_installed_pytest.sh tests/fast/test_iceberg_s3_multinode_live.py \
  -m 'external_service and real_ray and ray_cluster_owner'
```

The runtime image uses
`python:3.12-slim-trixie@sha256:a0f57064cbdcd6285fa45f51237ab0b35ea8ebaed06a1f5973e89e19dfb86704`.
It installs ordinary wheels offline and verifies `pip check` plus executable
permission on Ray's GCS binary. Preserve executable modes when preparing any
test-only reconstructed wheel cache. This run reconstructed its dependency
cache from installed files with RECORD hash verification, retaining notices,
metadata and executable modes; those cache wheels are not production artifacts.
The fixture SDK and writer credentials remain on the host, outside that image.
The profile pins security fixes in certifi 2026.10.19
([advisory](https://github.com/advisories/GHSA-6q48-6c7w-9gw2)), filelock 4.2.1
([advisory](https://github.com/advisories/GHSA-v4x8-3wr5-3wq8)) and urllib3 2.8.0
([chunk parsing](https://github.com/advisories/GHSA-vxq7-64xx-v4gw),
[proxy TLS](https://github.com/advisories/GHSA-8988-9cw3-xx77)).

`VANE_TEST_ICEBERG_DOCKER_COMMAND` optionally selects a command prefix, such as
`sudo -n docker` after separately authorizing sudo. Tests never request or store
a sudo password. `VANE_TEST_ICEBERG_MULTINODE_DIAGNOSTICS` selects an external
directory for sanitized topology, probe results, object-request audits, service
logs and cleanup evidence. Service logs are private files. Without an explicit
immutable runtime image, the suite skips before starting Docker or importing
the optional SDK.

Each test invocation owns a random Docker label/network and its exact
container IDs. Cleanup verifies ownership and removes only those resources,
including after a failed startup. Ray sizes its object store normally within
each container's 4 GiB memory limit and 2 GiB shared-memory limit; the harness
sets no object-store capacity override. Readiness, fault and cleanup waits are
bounded. This gate does not use a process-wide `ray stop` on the host.

The qualified boundary is Linux x86-64 with homogeneous containers on one
Docker host. Physical multi-host deployments, mixed operating systems,
transport security and representative throughput remain unqualified. Private
S3, authentication, credential vending/refresh, catalog writes and production
provider publication keep their separate gates.

### Remaining roadmap

| Stage | Delivery gate |
| --- | --- |
| M5a: frozen transport | Native restored execution, stable snapshot, detached worker bind and explicit unsupported-state guards |
| M5b: file assignments and Ray | Per-file assignments, same-host real worker execution, recovery, cancellation and exact provider replay; public Ray API for the narrow matrix above |
| M5c1: identity partitions | Owned per-file constants, omitted-column recovery, historical identity specs, real Ray replay and lifecycle checks as above |
| M5c2a: transformed partitions | Known time/bucket/truncate/void specs, original-value reads, mixed historical specs and real-Ray lifecycle checks above |
| M5c2b: schema evolution | Field-ID mapping for column add/drop/rename/reorder, supported type/default semantics and old-snapshot coverage |
| M5c3: deletes | Committed v2 position/equality descriptors, exact applicability, hidden field-ID mapping, NULL-safe matching and native/real-Ray lifecycle coverage above |
| M5d1: anonymous S3 | Canonical S3 data/delete locators, explicit connection-setting replay, real MinIO and local/single-node Ray qualification above |
| M5d2: isolated nodes | Driver/head plus two isolated Ray workers, S3-only dataset sharing, exact provider deployment, retry/rejoin and cleanup without dataset mounts as above |
| Later optional work | Production provider delivery; authentication/credential refresh and catalog writes remain separate |

The current fixture does not qualify v1/v3 tables, other remote filesystems,
physical multi-host deployment, catalog writes, authentication extensions, private S3,
credential vending or refresh.
