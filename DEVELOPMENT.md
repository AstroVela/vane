# Development

Vane contains Python, pybind11, and a modified DuckDB C++ engine. A native build also links Arrow Flight, gRPC, and selected DuckDB extensions.

## Prerequisites

- Linux x86-64 for the currently tested path
- Python 3.10, 3.11, or 3.12; Python 3.12 is recommended and is the primary development version
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
python -m pytest tests/fast
python -m pytest tests/slow
python -m pytest tests/ai
```

Some tests require network access, model weights, GPUs, credentials, or a local MinIO/Ray setup. Tests must skip with a clear reason when an optional environment is absent; they must not silently use a maintainer's local endpoint or credentials.

## Formatting and static checks

```bash
python -m pip install pre-commit
pre-commit install
scripts/format root --changed
python scripts/sync_duckdb_source_id.py --check
pre-commit run --from-ref origin/main --to-ref HEAD
```

Run `pre-commit install` once per clone. The installed commit hook repairs
`DUCKDB_SOURCE_ID` from the staged DuckDB tree; the explicit check above also
covers clean range-based runs, where the Git index has no staged changes.

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
as commit trailers. `scripts/format duckdb` and `scripts/format workspace`
automatically synchronize the content-derived identity after a successful
formatter pass. To synchronize or verify it explicitly, run:

```bash
python scripts/sync_duckdb_source_id.py
python scripts/sync_duckdb_source_id.py --check
```

The script records the full Git tree object in `DUCKDB_SOURCE_ID`, including
staged, unstaged, and untracked non-ignored engine files without changing the
real Git index. If formatting was skipped, pre-commit updates the file from the
staged DuckDB tree; stage that generated change and retry the commit. CI checks
the record before building source packages, including commits made with hooks
disabled. Update `SOURCE_PROVENANCE.md` and `OVERRIDE_GIT_DESCRIBE` only when
the imported upstream baseline, DuckDB version line, or historical mapping
changes.

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
