# DuckDB compatibility and versioning

Vane embeds a modified DuckDB engine, but it does not replace the official
Python `duckdb` distribution. Both distributions are supported in the same
Python environment and in the same process.

## Package ownership

The `vane-ai` distribution owns these import and type-stub names:

- `vane`
- `_vane_duckdb`
- `_vane_duckdb-stubs`
- `vane_adbc_driver_duckdb`

It does not install or register aliases for `duckdb`, `_duckdb`,
`_duckdb-stubs`, or `adbc_driver_duckdb`. Those names remain owned by the
official DuckDB distribution. Installing either distribution before the other,
or uninstalling either one, must leave the other distribution usable.

Applications select an engine explicitly:

```python
import duckdb  # Official DuckDB distribution
import vane    # Vane's private DuckDB fork
```

Vane re-exports much of DuckDB's Python API for source familiarity, but the two
modules contain distinct native engines and distinct Python types. Connections,
relations, expressions, and extension objects from one engine must not be passed
to the other unless an API explicitly documents an interchange format such as
Arrow.

The bundled ADBC facade is imported as `vane_adbc_driver_duckdb`. The official
`adbc_driver_duckdb` package, when installed, remains independent.

## Upgrading from `vane-ai 0.1.0a1`

The published `0.1.0a1` wheel predates the private package names above. Its pip
installation record claims files under `duckdb`, `_duckdb`, `_duckdb-stubs`,
and `adbc_driver_duckdb`. If that Vane release and official DuckDB are installed
together, a normal Vane upgrade can remove files still needed by DuckDB while
pip uninstalls the old wheel.

For an existing shared environment, remove both conflicting distributions and
then install the new versions into the clean namespace:

```bash
python -m pip uninstall -y vane-ai duckdb
python -m pip install duckdb vane-ai
```

If Vane has already been upgraded and `import duckdb` is broken, restore the
official package without changing the newly private Vane installation:

```bash
python -m pip install --force-reinstall duckdb
```

This one-time migration is required because a new wheel cannot change the file
ownership record embedded in an already-installed `0.1.0a1` wheel. Releases
after `0.1.0a1` have disjoint ownership and do not require this repair.

## Version contract

- `importlib.metadata.version("vane-ai")`, `vane.__version__`, and
  `vane.__vane_version__` are the Vane distribution version.
- `vane.__duckdb_version__` and `_vane_duckdb.__version__` are the version of
  the embedded DuckDB engine.
- `vane.__git_revision__` and `SELECT source_id FROM pragma_version()` identify
  Vane's exact embedded engine source tree as described in
  [SOURCE_PROVENANCE.md](SOURCE_PROVENANCE.md).
- `duckdb.__version__` belongs exclusively to the official DuckDB distribution
  and is never rewritten by Vane.

The Vane and DuckDB version numbers are intentionally independent. Code that
reports diagnostics should include both `vane.__version__` and
`vane.__duckdb_version__`.
