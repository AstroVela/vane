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

The engine's extension configuration pins Iceberg to
[`AstroVela/duckdb-iceberg@f83b867b9b4c5b4a1c7eefd61a16e46a8cf55b1c`](https://github.com/AstroVela/duckdb-iceberg/tree/f83b867b9b4c5b4a1c7eefd61a16e46a8cf55b1c).
Native engine compatibility is maintained in that repository and selected by
`ICEBERG_VANE_NATIVE=ON`. This qualification keeps
`ICEBERG_VANE_DISTRIBUTED=OFF`. It does not qualify write operations or Ray scans.
Vane fetches the pinned source directly; no compatibility patch is applied.
For an explicit source override, prepare a clean checkout of that commit:

```bash
git clone --no-checkout https://github.com/AstroVela/duckdb-iceberg.git /absolute/path/to/pinned-iceberg
git -C /absolute/path/to/pinned-iceberg checkout --detach f83b867b9b4c5b4a1c7eefd61a16e46a8cf55b1c
export DUCKDB_ICEBERG_DIRECTORY=/absolute/path/to/pinned-iceberg
```

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
