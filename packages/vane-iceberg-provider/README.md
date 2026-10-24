# Iceberg provider qualification SDK

This is the separate M4a development SDK profile from
[issue #1064](https://github.com/AstroVela/vane/issues/1064). It supplies the
native dependencies for Vane's source-pinned Iceberg and Avro providers.
Production provider publication is tracked separately in
[issue #619](https://github.com/AstroVela/vane/issues/619).

Follow the repository [development workflow](../../DEVELOPMENT.md) to prepare
the base SDK and a noneditable Python environment first. This profile keeps the
existing builtin baseline, adds the pinned DuckDB Avro C registry, and enables
only AWS SDK core/SSO/STS instead of its default service set. These are build
dependencies of the upstream provider; this qualification uses anonymous REST
and local files.

The Iceberg checkout must receive the Vane-owned
`patches/iceberg-vane-local.patch` before compilation. It targets exactly
`45163a28e0ed6a2071a82a1bf1dd432d0216cf9c`, including Vane's logical-write
constructor API. Carrying this patch does not depend on an upstream merge or
qualify MERGE/INSERT/UPDATE/DELETE behavior. Apply it once to a clean checkout:

```bash
git -C "$DUCKDB_ICEBERG_DIRECTORY" apply --check \
  "$PWD/packages/vane-iceberg-provider/patches/iceberg-vane-local.patch"
git -C "$DUCKDB_ICEBERG_DIRECTORY" apply \
  "$PWD/packages/vane-iceberg-provider/patches/iceberg-vane-local.patch"
git -C "$DUCKDB_ICEBERG_DIRECTORY" apply --check \
  "$PWD/packages/vane-iceberg-provider/patches/iceberg-vane-frozen-scan.patch"
git -C "$DUCKDB_ICEBERG_DIRECTORY" apply \
  "$PWD/packages/vane-iceberg-provider/patches/iceberg-vane-frozen-scan.patch"
git -C "$DUCKDB_ICEBERG_DIRECTORY" apply --check \
  "$PWD/packages/vane-iceberg-provider/patches/iceberg-vane-file-scan.patch"
git -C "$DUCKDB_ICEBERG_DIRECTORY" apply \
  "$PWD/packages/vane-iceberg-provider/patches/iceberg-vane-file-scan.patch"
git -C "$DUCKDB_ICEBERG_DIRECTORY" apply --check \
  "$PWD/packages/vane-iceberg-provider/patches/iceberg-vane-partitions.patch"
git -C "$DUCKDB_ICEBERG_DIRECTORY" apply \
  "$PWD/packages/vane-iceberg-provider/patches/iceberg-vane-partitions.patch"
git -C "$DUCKDB_ICEBERG_DIRECTORY" apply --check \
  "$PWD/packages/vane-iceberg-provider/patches/iceberg-vane-transforms.patch"
git -C "$DUCKDB_ICEBERG_DIRECTORY" apply \
  "$PWD/packages/vane-iceberg-provider/patches/iceberg-vane-transforms.patch"
git -C "$DUCKDB_ICEBERG_DIRECTORY" apply --check \
  "$PWD/packages/vane-iceberg-provider/patches/iceberg-vane-schema.patch"
git -C "$DUCKDB_ICEBERG_DIRECTORY" apply \
  "$PWD/packages/vane-iceberg-provider/patches/iceberg-vane-schema.patch"
```

The second patch adds M5a's versioned frozen-snapshot transport on top of the
same source pin. It supports only unpartitioned format-v2 local Parquet with
one flat schema and no delete manifests. The loadable Iceberg artifact links
the pinned Parquet implementation so it remains self-contained; no external
Parquet provider or `_native` private-symbol import is required. Rebuild both
Iceberg and Avro whenever the engine identity changes. The third patch replaces
M5a's whole-snapshot assignment with M5b's independently schedulable file
assignments. The fourth patch adds M5c1's owned identity partition constants,
including NULLs and per-file historical partition specs. It shares the pinned
Iceberg decoder with the live reader and bumps the frozen bind/file codec to
version 2. The fifth patch admits M5c2a's known transformed partition specs while
freezing only identity constants. Non-identity values never substitute source
column values. The wire structure stays at version 2. Keep all five patches in
this order when reproducing M5c2a. The sixth patch admits M5c2b's flat schema
history and fixes legacy partition decoding/filter lookups after a source is
dropped. Names, order, field IDs, target types and typed NULL defaults remain
owned version-2 state. Keep all six patches in order for the current build;
the five predecessor patches are unchanged.
See [the schema qualification and roadmap](../../GRAVITINO_ICEBERG.md#flat-schema-evolution-m5c2b)
for the supported table semantics and shared-filesystem boundary.
Provider-wheel packaging must retain the DuckDB and vendored dependency notice
inputs listed in the root `pyproject.toml`'s `license-files`, including Parquet's
dependencies, together with the pinned Iceberg/Avro and SDK notices.

From the repository root, using the baseline-matched vcpkg tool checkout:

```bash
"$VCPKG_ROOT/vcpkg" install \
  --x-manifest-root="$PWD/packages/vane-iceberg-provider" \
  --x-install-root="$PWD/build/iceberg-sdk" \
  --triplet=x64-linux-release --host-triplet=x64-linux-release \
  --clean-buildtrees-after-build --clean-packages-after-build

python -I scripts/check_copyleft.py \
  --manifest packages/vane-iceberg-provider/vcpkg.json \
  --share-dir build/iceberg-sdk/x64-linux-release/share
python scripts/sync_vcpkg_licenses.py \
  --share-dir build/iceberg-sdk/x64-linux-release/share \
  --output LICENSES/iceberg-native-dependencies.txt --check
```

The committed notice bundle records the qualified Linux dependency profile.
Other platforms or dependency changes require their own notice comparison and
qualification. Liblzma 5.8.3's linked library uses 0BSD; its complete retained
notice also describes separately licensed XZ programs. SDK executables are not
provider wheel contents.

`requirements-test.txt` installs the independent fixture writer and reader;
those Python packages are not Vane runtime dependencies. The fixture explicitly
selects PyIceberg's existing `noop` auth mode and bypasses named catalog
configuration from a developer's home directory.

Avro must be loaded before Iceberg. Package both artifacts with the existing
signed test-provider workflow and declare the exact Avro wheel as Iceberg's
dependency. Calling `vane.load_installed_extension("iceberg", connection=con)`
then loads and validates the complete local dependency closure. Keep test
signing keys and `--test-only` artifacts out of releases.
