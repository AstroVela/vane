# AI Agent Guidelines

Follow [DEVELOPMENT.md](DEVELOPMENT.md) for the development workflow. Keep
checkout-specific build, test, and release instructions in this repository.
The [published Development Guide](https://vane.astrovela.ai/docs/data/contributing/development)
provides contributor tutorials; link between the guides instead of duplicating
their full contents.

## Build

Do not use an editable install. Python-only changes do not require a native rebuild. After changing C++, reinstall using the incremental build directory:

```bash
export SKBUILD_BUILD_DIR="$PWD/build/python-release"
export SKBUILD_CMAKE_BUILD_TYPE=Release
uv pip install . --no-build-isolation
```

Do not commit the generated `DUCKDB_FORK_REVISION` or `DUCKDB_SOURCE_ID`
manifests. See [engine identities](DEVELOPMENT.md#updating-the-duckdb-subtree)
for inspection commands and custom source-tree requirements.

## Formatting

```bash
scripts/format root --changed
scripts/format duckdb --changed
scripts/format workspace --changed
```

Use `root` for Vane-owned files and `duckdb` for the `external/duckdb` subtree. Use `workspace` only when both contain changes.

## Tests

Run the tests affected by the change first, then run the Vane base test suite:

```bash
scripts/run_installed_pytest.sh tests/fast/test_udf_process.py
scripts/run_release_tests.sh
```

To run the complete fast Python test suite:

```bash
scripts/run_fast_tests.sh
```

The launcher runs non-Ray tests, shared-cluster Ray tests, and test-owned Ray
clusters in separate pytest processes. Do not replace it with one long-lived
`pytest tests/fast` process.

The fast/release Ray shards let Ray size the object store from the node's
available memory by default. Use `VANE_TEST_RAY_OBJECT_STORE_BYTES` only to pin
the capacity for a specialized test; it does not configure production clusters.
Tests that call `ray.init()` directly must be marked `real_ray` and
`ray_cluster_owner`. Tests that require CUDA hardware must also be marked `gpu`;
the standard CPU-only CI fast-test shards exclude that marker.
