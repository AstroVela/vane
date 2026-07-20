# AI Agent Guidelines

Follow [DEVELOPMENT.md](DEVELOPMENT.md) for the development workflow. The
[published Development Guide](https://vane.astrovela.ai/docs/data/contributing/development)
should mirror that file.

## Build

Do not use an editable install. Python-only changes do not require a native rebuild. After changing C++, reinstall using the incremental build directory:

```bash
export SKBUILD_BUILD_DIR="$PWD/build/python-release"
export SKBUILD_CMAKE_BUILD_TYPE=Release
uv pip install . --no-build-isolation
```

Native builds compute the content-derived DuckDB SourceID without modifying the
checkout. Direct incremental builds watch the external tree for CMake
reconfiguration and refresh a generated header in the build directory. The
engine version object and default in-tree static extension entry points consume
that header, including after mode-only changes. Git-exported trees derive the
same Git-compatible identity from their files when no manifest exists. The PEP
517 backend injects `DUCKDB_SOURCE_ID` into source distributions; do not add
that generated file to Git.

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
python -m pytest tests/fast/test_udf_process.py
scripts/run_release_tests.sh
```

To run the complete fast Python test suite:

```bash
python -m pytest tests/fast
```
