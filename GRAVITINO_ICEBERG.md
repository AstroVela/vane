# Gravitino Iceberg provider qualification

M4a tracks an independently built Iceberg provider and native local reads
through Gravitino's Iceberg REST service. The tracking issue is
[#1064](https://github.com/AstroVela/vane/issues/1064); production wheel
publication remains [#619](https://github.com/AstroVela/vane/issues/619).
This qualification does not introduce a stable Python table-read API or change
M3's general Iceberg capability flags.

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

## Compatibility changes and next milestones

The pristine Iceberg build fails in its MERGE planner because Vane's logical
INSERT/UPDATE/DELETE constructors require `ClientContext`. The compatibility
patch passes the context already owned by that planner. Successful compilation
does not qualify any write operation; acceptance attaches READ_ONLY.

Upstream `IcebergScanSerialize` and `IcebergScanDeserialize` explicitly throw
`NotImplementedException`. Vane's native local path executes the live plan
directly. Ray requires an Iceberg-owned transport and split protocol; inherited
Parquet callbacks cannot establish that contract for Iceberg bind state.

M4b should qualify the stable local read interface and its feature matrix,
including time travel, partitions, schema evolution and delete files. M5 must
freeze a snapshot per execution, serialize the relevant Iceberg state, assign
scan splits, and preserve field IDs and deletion semantics on workers. Each
unsupported feature must fail explicitly. Vane can implement these adapters
against the fixed source without waiting for upstream changes.

This fixture does not qualify v1/v3 tables, production object stores, distributed
reads, catalog writes, authentication extensions, credential vending or refresh.
