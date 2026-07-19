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

DuckDB and workspace formatting automatically synchronize the content-derived
`DUCKDB_SOURCE_ID`. The pre-commit hook also repairs it from the staged DuckDB
tree if formatting was skipped; install that hook once per clone with
`pre-commit install`. CI rejects an out-of-date value.

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
