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

Python-only changes do not require a native rebuild. Changes below `src/duckdb_py/` or `external/duckdb/src/` do.

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

## Python tests

The required release gate covers the supported base installation and does not
need model downloads, cloud credentials, GPUs, or external services:

```bash
scripts/run_release_tests.sh
```

The inherited compatibility suites are broader and require the development
dependency group. Run them when changing the corresponding integration:

```bash
scripts/run_fast_tests.sh
python -m pytest tests/slow
python -m pytest tests/ai
```

The fast-test launcher runs non-Ray tests, shared-cluster Ray tests, and
test-owned Ray clusters in separate pytest processes. This keeps the real Ray
runtime out of the long-lived non-Ray pytest process. Fast and release Ray test
clusters use a 2 GiB object store by default;
`VANE_TEST_RAY_OBJECT_STORE_BYTES` overrides this test-only profile and does
not configure production clusters. Tests that call `ray.init()` directly must
be marked `real_ray` and `ray_cluster_owner`.

CI further splits the non-Ray phase across CPU-only jobs. The jobs install the
built wheel, use CPU-only PyTorch, cap Ray task heap requests at 1 GiB, and set
hard pytest-process and job deadlines so the suite fits a standard 4-vCPU,
16-GiB GitHub-hosted runner. Tests marked `gpu` are excluded there because
standard runners do not provide CUDA hardware; run the default launcher on a
GPU host to include them.

Tests that require an externally provisioned service are excluded by default.
Run them explicitly when the required service and credentials are available:

```bash
python -m pytest -m external_service tests/fast
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
as commit trailers. To inspect the content-derived identity without writing the
checkout, run:

```bash
python scripts/sync_duckdb_source_id.py --print
```

The script computes the full Git tree object for `external/duckdb`, including
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
runtime SourceID on the first incremental build. The local PEP 517 backend
injects `DUCKDB_SOURCE_ID` directly into the completed sdist, so read-only source
trees remain supported.
The sdist carries that manifest for subsequent builds without Git metadata, and
artifact validation checks it against the checkout. The manifest is ignored
build metadata and must not be committed, so parallel engine pull requests do
not modify a shared generated file. Update `SOURCE_PROVENANCE.md` and
`OVERRIDE_GIT_DESCRIBE` only when the imported upstream baseline, DuckDB version
line, or historical mapping changes.

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
