# Development

Vane contains Python, pybind11, and a modified DuckDB C++ engine. A native build also links Arrow Flight, gRPC, and selected DuckDB extensions.

## Prerequisites

- Linux x86-64 for the complete build and test path
- macOS arm64 and Windows x86-64 for the native build and distributed-test paths used by CI
- Python 3.10 through 3.14; Python 3.12 is recommended and is the primary development version
- Git with `git subtree` support
- A C++20 compiler and CMake 3.29+; Ninja and ccache on Linux/macOS, or Visual Studio 2022 on Windows
- vcpkg at the baseline pinned in `vcpkg.json`

The DuckDB engine fork is included directly under `external/duckdb`; a normal
clone contains all source needed for the build.

Bootstrap native dependencies from the repository root:

```bash
bash scripts/bootstrap_vcpkg.sh
```

The helper checks out the exact baseline from `vcpkg.json`, installs into
`vcpkg_installed`, and verifies the committed native-dependency license bundle.
It selects the host platform's release-only target and host triplets by default,
including `x64-linux-release` on Linux x86-64, `arm64-osx-release` on Apple
Silicon, and `x64-windows-static-release` on Windows x86-64. Set
`VCPKG_TARGET_TRIPLET=x64-linux` when both release and debug target dependency
builds are needed; `VCPKG_HOST_TRIPLET` independently overrides the host tools
triplet. CMake selects only the requested or platform-default triplet, without
searching other installed triplets. Set `VCPKG_INSTALLED_DIR` to select an
alternative dependency installation for bootstrap, CMake, and license tools.
Relative installation paths are resolved from the repository root.
When intentionally changing native dependencies, regenerate the bundle
with `python scripts/sync_vcpkg_licenses.py` and review its diff. Successful port
builds are cached before their temporary build and package trees are removed,
keeping bootstrap within hosted-runner disk limits.

Run `python -I scripts/check_copyleft.py` after source or dependency changes.
The bootstrap also checks installed GPL-family notices against the reviewed
inventory. Follow [COPYLEFT.md](COPYLEFT.md) before updating license hashes,
selecting a different license alternative, or adding codec features.
When checking an optional native dependency tree, pass `--share-dir <share>`
and repeat `--feature <vcpkg-feature>` for every selected feature so missing
required transitive notices are rejected as well.

## Incremental package build

Create and activate a virtual environment, then reuse a persistent native build directory:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

export SKBUILD_BUILD_DIR="$PWD/build/python-release"
export SKBUILD_CMAKE_BUILD_TYPE=Release
python -m pip install . --no-build-isolation -v
```

Do not use `pip install -e`. An editable install can cause Ray workers to invoke the build backend during import and delay actor startup.

Python-only changes do not require a native rebuild, but reinstall the
non-editable package so the test environment receives them. Changes below
`src/vane_py/` or `external/duckdb/src/` require an incremental native build.

## Building a loadable extension artifact

For `native_media`, first prepare its separate SDK and shared libraries using
[the media build guide](NATIVE_MEDIA_EXTENSIONS.md#build-and-package).
Run `tests/fast/test_ray_native_runtime_replacement.py` separately from
shared-cluster tests with the signed provider fixture; it owns two-node clusters.

`VANE_LOADABLE_EXTENSIONS` builds selected DuckDB extensions as self-contained
`.duckdb_extension` artifacts without linking them into `vane._native`. The
default is empty, so base Vane builds and wheels do not contain staged optional
extensions. DuckDB's pinned source configuration is preserved for external
extensions such as `httpfs`. For example, build and exercise both the in-tree
`tpch` artifact and the externally sourced `httpfs` artifact:

```bash
export SKBUILD_BUILD_DIR="$PWD/build/python-release"
export SKBUILD_CMAKE_BUILD_TYPE=Release
uv pip install . --no-build-isolation \
  '-Ccmake.define.VANE_LOADABLE_EXTENSIONS=tpch;httpfs'
cmake --build "$SKBUILD_BUILD_DIR" --target vane_loadable_extensions
VANE_TEST_LOADABLE_EXTENSION_PATH=\
"$SKBUILD_BUILD_DIR/vane_extensions/tpch.duckdb_extension" \
VANE_TEST_LOADABLE_HTTPFS_EXTENSION_PATH=\
"$SKBUILD_BUILD_DIR/vane_extensions/httpfs.duckdb_extension" \
  scripts/run_installed_pytest.sh tests/fast/test_loadable_extension_artifacts.py
```

Loadable artifacts require `EXTENSION_STATIC_BUILD=ON`. This keeps each
artifact self-contained and preserves Vane's private `_native` symbol boundary.
The staging directory is configurable with
`VANE_LOADABLE_EXTENSION_OUTPUT_DIRECTORY`.

For the independently selectable image, audio, and video operators, see
[NATIVE_MEDIA_EXTENSIONS.md](NATIVE_MEDIA_EXTENSIONS.md).

## Building an optional extension wheel

Keep optional artifacts out of the base `vane-ai` wheel. After installing Vane
and staging a release-approved extension, package it with:

```bash
python -I scripts/build_extension_wheel.py \
  --artifact "$SKBUILD_BUILD_DIR/vane_extensions/<extension>.duckdb_extension" \
  --extension-name <extension> \
  --platform-tag manylinux_2_28_x86_64 \
  --trust-identity astrovela/vane \
  --license-expression "Apache-2.0 AND MIT" \
  --license-file LICENSE \
  --license-file NOTICE \
  --license-file LICENSES/DuckDB-MIT.txt \
  --output-directory dist/extensions
```

Replace the example platform and license expression with those of the actual
artifact, and supply all required notices. The builder checks binary platform
requirements, package metadata, source identity, signatures and dependency
closure. It pins the matching Vane version and emits a content-addressed provider.
Pass the complete dependency closure in load order with repeated
`--dependency-wheel` arguments and explicitly allowlist each unique signer with
`--dependency-trust-identity`.

Verify the exact base and provider wheels in a clean environment on the target
platform's minimum supported runtime:

```bash
python -I scripts/verify_extension_wheel.py \
  --base-wheel dist/vane_ai-*.whl \
  --extension-wheel dist/extensions/vane_extension_<extension>-*.whl \
  --extension-name <extension> \
  --trust-identity astrovela/vane
```

Supply the same dependency closure and signer allowlist to verification.
`tpch` is an in-tree test artifact and must not be published as an extension
wheel. Test fixtures may use `--test-only`; they must never be published.

For specialized workflows and their contracts, see:

- [Native media packaging](NATIVE_MEDIA_EXTENSIONS.md#build-and-package):
  dynamic runtime/source SDK inputs and static release materials.
- [Media publication](NATIVE_MEDIA_RELEASE.md) and
  [library replacement](NATIVE_MEDIA_REPLACEMENT.md): delivery and acceptance.
- [Distributed extensions](DISTRIBUTED_EXTENSIONS.md#build-and-loading):
  provider discovery, exact artifact identity and Ray worker preparation.
- [Release process](RELEASE.md): production and TestPyPI signing policies.
- [Copyleft inventory](COPYLEFT.md): reviewed dependencies and source delivery.

The public CI signing key is enabled only by
`VANE_ENABLE_TEST_EXTENSION_SIGNING_KEY`; never enable it for release artifacts.
No-tag TestPyPI candidates instead use
`VANE_ENABLE_TESTPYPI_EXTENSION_SIGNING_KEY`. Production signing belongs only
in protected release-signing jobs; promote the same signed bytes after
qualification. Never bypass signature verification to load an incompatible
provider.

The validators in `vane_packaging/` define archive bounds and ELF, Mach-O and PE
checks. Keep policy changes with their regression tests. The manylinux policy
is an unmodified auditwheel snapshot: update its pinned version, digest and
license together in `vane_packaging/manylinux_policy.py` and `_vendor/auditwheel/`,
then run the policy parity and Linux extension-wheel tests. Do not edit the
snapshot to admit an otherwise incompatible artifact.

## Native C++ tests

Vane, DuckDB, and the non-Arrow distributed engine build as C++11, while the
Arrow Flight exchange, its direct tests, and the diagnostics boundary tests use
C++20. The diagnostics tests link C++20 callers against C++11 engine definitions.
This keeps Arrow's requirement isolated from the engine and its consumers. The
script refreshes the CMake configuration (`cmake --fresh`) to avoid configuration
drift and reuses compiled objects whose inputs are unchanged:

```bash
scripts/run_native_tests.sh "[distributed]"
```

The optional `native_media` extension requires C++17 for media reader construction
and exact video time arithmetic. Both its static and loadable targets keep this
requirement private, so it does not change the engine's language standard.
The native suite also links C++17 references to logical type constants against
their single exported definitions in the C++11 engine.

Run a named engine test or the complete unit suite with the same build:

```bash
scripts/run_native_tests.sh "test name" -s
scripts/run_native_tests.sh
```

The build uses two parallel compile jobs by default to stay within standard CI
runner memory. Override that limit with `VANE_NATIVE_BUILD_JOBS` when the local
machine has more capacity.

The launcher uses Ninja's single Release configuration by default. Windows CI
disables Git's automatic CRLF conversion before checkout so source-license
hashes and DuckDB SourceID use the committed bytes. Windows source checkouts
must likewise use `core.autocrlf=false` before files are checked out.
The Windows job follows DuckDB's native MSVC path with
`VANE_NATIVE_CMAKE_GENERATOR="Visual Studio 17 2022"` and
`VANE_NATIVE_CMAKE_GENERATOR_PLATFORM=x64`; the launcher restricts the
multi-config build to Release and runs the corresponding test executable. It
uses the pinned vcpkg toolchain in classic mode so Windows package wrappers
resolve release-only static library names without installing dependencies a
second time.

Statically linked DuckDB extensions participate in Ray execution through the
explicit scan callback and write provider contracts described in
[DISTRIBUTED_EXTENSIONS.md](DISTRIBUTED_EXTENSIONS.md). Add engine-level
protocol tests and extension-specific normal and fault-tolerant tests when
implementing either contract.

## Python tests

The required release gate covers the supported base installation and does not
need model downloads, cloud credentials, GPUs, or external services:

```bash
scripts/run_release_tests.sh
```

Vane's native extension is private to the installed `vane` package. Test
launchers therefore run outside the checkout and put the installed
site-packages directory before repository support modules. This prevents the
source package from shadowing `vane._native` and ensures tests exercise the
same layout shipped in the wheel. Run an affected test through the same
installed-package wrapper:

```bash
scripts/run_installed_pytest.sh tests/fast/test_udf_process.py
```

The inherited compatibility suites are broader and require the development
dependency group. Run them when changing the corresponding integration:

```bash
scripts/run_fast_tests.sh
scripts/run_installed_pytest.sh tests/slow
scripts/run_installed_pytest.sh tests/ai
```

The fast-test launcher runs non-Ray tests, shared-cluster Ray tests, and
test-owned Ray clusters in separate pytest processes. This keeps the real Ray
runtime out of the long-lived non-Ray pytest process. Fast and release Ray test
clusters let Ray size the object store from the node's available memory by
default. `VANE_TEST_RAY_OBJECT_STORE_BYTES` pins the capacity for a specialized
test; it does not configure production clusters. Tests that call `ray.init()`
directly must be marked `real_ray` and `ray_cluster_owner`.

CI further splits the non-Ray phase across CPU-only jobs. The jobs install the
built wheel, use CPU-only PyTorch, and set hard pytest-process and job deadlines
so the suite fits a standard 4-vCPU, 16-GiB GitHub-hosted runner. Tests marked
`gpu` are excluded there because standard runners do not provide CUDA hardware;
run the default launcher on a GPU host to include them.

The Linux native-build CI job applies `.github/ci-constraints.txt` to every pip
install, including wheel reinstalls. It temporarily uses Ray 2.56.0 on Python
3.14 because Ray 2.57/2.58 can abort concurrent async actors
([Ray #66399](https://github.com/ray-project/ray/issues/66399)). All tests remain
enabled. This is a CI workaround, not a production recommendation: older Ray
has a Python 3.14 async-actor memory leak
([Ray #63290](https://github.com/ray-project/ray/issues/63290)). Use Python 3.12
or 3.13 for long-running Ray workloads until an upstream fix is available.
Remove the constraint once a fixed Ray release passes the Python 3.14 media
concurrency tests and the base release suite.

Tests that require an externally provisioned service are excluded by default.
Run them explicitly when the required service and credentials are available:

```bash
scripts/run_installed_pytest.sh -m external_service tests/fast
```

Other optional tests may require network access, model weights, GPUs, credentials, or a local Ray setup. Tests must
skip with a clear reason when an optional environment is absent; they must not silently use a maintainer's local
endpoint or credentials.

## Formatting and static checks

```bash
python -m pip install pre-commit
pre-commit install
scripts/format root --changed
pre-commit run --from-ref origin/main --to-ref HEAD
```

Run `pre-commit install` once per clone.

Add `--check` to verify formatting without modifying files. Use `workspace`
when both Vane-owned files and the DuckDB subtree have changed:

```bash
scripts/format workspace --changed --check
```

To check changes relative to a committed ref, including in CI, use:

```bash
scripts/format workspace --from-ref origin/main --check
```

The root formatter deliberately excludes `external/duckdb`. Format DuckDB subtree changes with:

```bash
scripts/format duckdb --changed
```

## Updating the DuckDB subtree

The official engine baseline is imported from `duckdb/duckdb` as a squashed
subtree snapshot. Pull a reviewed upstream revision using the same mode:

```bash
git subtree pull --prefix=external/duckdb --squash \
  https://github.com/duckdb/duckdb.git main
```

The subtree metadata records the exact official DuckDB revision in
`git-subtree-split`. Vane-specific engine changes live as subsequent commits
under `external/duckdb`; review and resolve them when updating the official
baseline. When replaying a change formerly maintained in another repository,
preserve its author and date and record the original commit and upstream parent
as commit trailers. To inspect both engine identities without writing the
checkout, run:

```bash
python scripts/sync_duckdb_source_id.py --print
python scripts/resolve_duckdb_fork_version.py --print-version
```

`SourceID` identifies the contents of `external/duckdb`, including non-ignored
untracked files and file modes. The fork version identifies the last Vane
commit that changed that subtree, prefixed by `DUCKDB_UPSTREAM_VERSION` and
suffixed with `-dirty` for uncommitted subtree changes. Both commands are read-only.
Incremental builds refresh these identities automatically, including mode-only
changes; generated headers live in the build directory.

Source distributions carry generated `DUCKDB_SOURCE_ID` and
`DUCKDB_FORK_REVISION` manifests. Do not commit them. Git-exported trees can
derive a Git-compatible SourceID from their files, but an archive without Git
history still requires the fork revision manifest. Update `SOURCE_PROVENANCE.md`
and `DUCKDB_UPSTREAM_VERSION` only for baseline, version or provenance changes.

A custom `DUCKDB_SOURCE_PATH` requires explicit full `VANE_DUCKDB_SOURCE_ID`
and `VANE_DUCKDB_FORK_REVISION` values, plus `VANE_DUCKDB_UPSTREAM_VERSION`
in `vX.Y.Z` form. Configuration fails if any is absent.

The original upstream history remains in `duckdb/duckdb`. Vane's path history
begins at the squashed snapshot and includes every later Vane engine commit. To
inspect or export that history with DuckDB-rooted paths, split it to a temporary
branch:

```bash
git subtree split --prefix=external/duckdb --ignore-joins -b duckdb-history
git log --stat duckdb-history
```

`--ignore-joins` produces a self-contained compact history containing the
official snapshot and Vane's subsequent commits. To reconnect the split branch
to DuckDB's complete upstream history instead, fetch `duckdb/duckdb` first and
omit `--ignore-joins`; Git uses the recorded `git-subtree-split` revision as the
join point.

## Debugging Ray workers

GPU batch UDFs on persistent Ray Actors adapt their compute batch size inside
the actor. A byte-bounded transport envelope can contain several compute
batches, so a small `batch_size` does not force one Ray submission per model
call. The actor reuses the native dynamic batch controller and retains its
history across envelopes. It times synchronous callable and result-generator
work, excluding queueing, actor initialization, and downstream output
backpressure. Short envelope tails are processed immediately; fast tails do not
lower the controller's capacity estimate, while slow tails can still trigger
smaller batches. Ray Task UDFs keep scheduler-side sizing but report the same
worker-measured callable/generator time in a final control pair. Task scheduling,
worker setup, input/output transport, and downstream backpressure do not enter
the controller's latency observations. Missing task timing is an error rather
than a fallback to submission-to-completion wall time.

A supplied GPU `batch_size` seeds the initial compute size, capped at 256
rows. It does not cap growth:
the separate dynamic upper bound is 131072 rows, and existing transport byte
limits still apply. The adjustment constants are unchanged.

Set `DUCKDB_DISTRIBUTED_DEBUG=1`. Native debug output uses `DistributedDebugStream()` and appears in Ray worker error logs, normally below `/tmp/ray/session_latest/logs/worker-*.err`. Plain C `stdout` output is not reliably captured by Ray workers.

## Release artifacts

Build and validate an sdist before opening a release pull request:

```bash
python -m build --sdist
python scripts/check_release_artifacts.py dist/*.tar.gz
```

See [RELEASE.md](RELEASE.md) for the complete process.
