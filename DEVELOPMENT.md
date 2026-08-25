# Development

Vane contains Python, pybind11, and a modified DuckDB C++ engine. A native build also links Arrow Flight, gRPC, and selected DuckDB extensions.

## Prerequisites

- Linux x86-64 for the currently tested path
- Python 3.10 through 3.14; Python 3.12 is recommended and is the primary development version
- Git with `git subtree` support
- A C++20 compiler, CMake 3.29+, Ninja, and ccache
- vcpkg at the baseline pinned in `vcpkg.json`

The DuckDB engine fork is included directly under `external/duckdb`; a normal
clone contains all source needed for the build.

Bootstrap native dependencies from the repository root:

```bash
bash scripts/bootstrap_vcpkg.sh
```

The helper checks out the exact baseline from `vcpkg.json`, installs into
`vcpkg_installed`, and verifies the committed native-dependency license bundle.
When intentionally changing native dependencies, regenerate the bundle with
`python scripts/sync_vcpkg_licenses.py` and review its diff.

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
`VANE_LOADABLE_EXTENSION_OUTPUT_DIRECTORY`; packaging and trusted artifact
metadata are intentionally handled separately.

## Native C++ tests

The complete native gate builds DuckDB, distributed exchange, and the test
runner with the same pinned Arrow and C++20 configuration used by CI. The
script starts from a fresh CMake configuration (`cmake --fresh`) to avoid
configuration drift, which triggers a clean rebuild in its build directory:

```bash
scripts/run_native_tests.sh "[distributed]"
```

Run a named engine test or the complete unit suite with the same build:

```bash
scripts/run_native_tests.sh "test name" -s
scripts/run_native_tests.sh
```

The build uses two parallel compile jobs by default to stay within standard CI
runner memory. Override that limit with `VANE_NATIVE_BUILD_JOBS` when the local
machine has more capacity.

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

The first command computes the full Git tree object for `external/duckdb`, including
staged, unstaged, and untracked non-ignored engine files without changing the
real Git index or object store. When Git metadata and a source-distribution
manifest are both absent, as in a `git archive` or GitHub source archive, the
script derives a Git-compatible tree object directly from the materialized
paths, modes, symlinks, and contents. Git expands the constant
`.git_archival.txt` template on export so the fallback preserves the
repository's SHA-1 or SHA-256 object format without a per-change identity file.
Native configuration registers the external tree as a CMake configuration
dependency, so Ninja and Makefile builds refresh configure-time metadata after
timestamp-visible source changes. A lightweight build target also refreshes a
generated header in the CMake binary directory. DuckDB's version object and the
entry points of its default in-tree static extensions force-include that header,
so mode-only changes that leave file timestamps untouched still update every
runtime SourceID on the first incremental build.

The second command reports the user-facing fork version as
`vX.Y.Z-vane.<revision>`. `vX.Y.Z` comes from `DUCKDB_UPSTREAM_VERSION`, and the
ten-character revision is calculated from the last Vane commit that changed
`external/duckdb`. Uncommitted changes within that directory append `-dirty`;
changes elsewhere in the checkout do not. Direct incremental builds refresh
the generated version header on every build, so committing an unchanged dirty
tree also replaces the dirty marker with the new path-changing commit.

A custom `DUCKDB_SOURCE_PATH` has no in-tree baseline to infer. Such builds must
set full `VANE_DUCKDB_SOURCE_ID` and `VANE_DUCKDB_FORK_REVISION` values and an
exact `VANE_DUCKDB_UPSTREAM_VERSION` in `vX.Y.Z` form. Configuration fails when
any of these explicit identities is absent; it never reuses the in-tree base.

The local PEP 517 backend injects full `DUCKDB_SOURCE_ID` and
`DUCKDB_FORK_REVISION` manifests directly into the completed sdist, so
read-only source trees remain supported. The sdist carries both manifests for
subsequent builds without Git metadata, and artifact validation checks them
against the checkout. The manifests are ignored build metadata and must not be
committed, so parallel engine pull requests do not modify shared generated
files. A source archive without Git history must contain the injected fork
revision manifest. Update `SOURCE_PROVENANCE.md` and
`DUCKDB_UPSTREAM_VERSION` only when the imported upstream baseline, DuckDB
version line, or historical mapping changes.

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

Set `DUCKDB_DISTRIBUTED_DEBUG=1`. Native debug output uses `DistributedDebugStream()` and appears in Ray worker error logs, normally below `/tmp/ray/session_latest/logs/worker-*.err`. Plain C `stdout` output is not reliably captured by Ray workers.

## Release artifacts

Build and validate an sdist before opening a release pull request:

```bash
python -m build --sdist
python scripts/check_release_artifacts.py dist/*.tar.gz
```

See [RELEASE.md](RELEASE.md) for the complete process.
