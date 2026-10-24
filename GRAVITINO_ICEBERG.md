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
[#1077](https://github.com/AstroVela/vane/issues/1077).
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
`iceberg-vane-partitions.patch` and `iceberg-vane-transforms.patch` from the
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
rejected. Timestamp-based selection and snapshot/schema-evolution combinations
are not part of this interface's qualified matrix.

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
is `cef38197067271d2e5c4d5f9da880cb184d2ec916fb795d366e43f78c4d0f789`.
The four earlier patches, source/SDK pins, engine identities and dependency
closure are unchanged. Only the Vane-owned frozen adapter is modified; no
additional Local/Ray runner implementation is introduced.

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
parser rejection. General schema evolution/name mapping and delete manifests
remain rejected by the frozen adapter.

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

### Remaining roadmap

| Stage | Delivery gate |
| --- | --- |
| M5a: frozen transport | Native restored execution, stable snapshot, detached worker bind and explicit unsupported-state guards |
| M5b: file assignments and Ray | Per-file assignments, same-host real worker execution, recovery, cancellation and exact provider replay; public Ray API for the narrow matrix above |
| M5c1: identity partitions | Owned per-file constants, omitted-column recovery, historical identity specs, real Ray replay and lifecycle checks as above |
| M5c2a: transformed partitions | Known time/bucket/truncate/void specs, original-value reads, mixed historical specs and real-Ray lifecycle checks above |
| M5c2b: schema evolution | Field-ID mapping for column add/drop/rename/reorder, supported type/default semantics and old-snapshot coverage |
| M5c3: deletes | Freeze position/equality delete state with correct file/sequence/partition applicability, NULL matching and retry semantics |
| Later optional work | Remote shared-storage qualification and production provider delivery; authentication/credential refresh and catalog writes remain separate |

The current fixture does not qualify v1/v3 tables, remote storage, catalog
writes, authentication extensions, credential vending or refresh.
